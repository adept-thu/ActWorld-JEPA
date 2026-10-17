"""Render an offline NAVSIM v1 submission replay; never generates model predictions.

Run in the NAVSIM v1 environment. Only use trusted dataset/submission pickles.
The camera and annotated actors are recorded context, not model perception.
Input clips JSON: [{"log": "name.pkl", "start": 10, "end": 30}, ...].
"""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw, ImageFont
from pyquaternion import Quaternion
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['submission', 'logs', 'sensors', 'maps', 'clips', 'output']:
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--map-version', default='nuplan-maps-v1.0')
    p.add_argument('--model-label', default='ActWorld-JEPA / R383')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep map locks in this render's output; do not write into shared datasets.
    from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB, MAP_DIMENSIONS
    from nuplan.database.common.blob_store.local_store import LocalStore
    def readonly_map_init(self, map_version, map_root):
        self._map_version = map_version
        self._map_root = map_root
        self._blob_store = LocalStore(map_root)
        self._metadata = json.loads((Path(map_root)/(map_version+'.json')).read_text())
        self._map_dimensions = MAP_DIMENSIONS
        self._max_attempts = 1
        self._seconds_between_attempts = 0
        self._map_lock_dir = str(args.output/'map_locks')
        Path(self._map_lock_dir).mkdir(exist_ok=True)
    GPKGMapsDB.__init__ = readonly_map_init
    with args.submission.open('rb') as f:
        submission = pickle.load(f)
    assert len(submission['predictions']) == 1
    predictions = submission['predictions'][0]
    clips = json.loads(args.clips.read_text())
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = lambda size: ImageFont.truetype(font_path, size)
    writer = cv2.VideoWriter(str(args.output/'navsim_replay.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), 2, (1600, 900))
    if not writer.isOpened():
        raise RuntimeError('OpenCV MPEG-4 encoder unavailable')
    records = []
    maps = {}
    try:
        for clip_index, clip in enumerate(clips):
            with (args.logs/clip['log']).open('rb') as f:
                frames = pickle.load(f)
            previous_time = None
            for index in range(clip['start'], clip['end']+1):
                frame = frames[index]
                if frame['token'] not in predictions:
                    raise ValueError('Missing prediction; refuse to interpolate it')
                timestamp = int(frame['timestamp'])
                if previous_time is not None and not 450000 <= timestamp-previous_time <= 550000:
                    raise ValueError('Clip is not a continuous 2 Hz sequence')
                previous_time = timestamp
                trajectory = np.asarray(predictions[frame['token']].poses)
                assert trajectory.shape == (8, 3) and np.isfinite(trajectory).all()
                image = Image.open(args.sensors/frame['cams']['CAM_F0']['data_path']).convert('RGB')
                # Conservative anonymization of likely plates/faces before publishing.
                rgb = np.asarray(image).copy()
                cam = frame['cams']['CAM_F0']
                for box, name in zip(frame['anns']['gt_boxes'], frame['anns']['gt_names']):
                    if str(name) not in ['vehicle', 'pedestrian', 'bicycle']:
                        continue
                    signs=np.array([[a,b,c] for a in [-1,1] for b in [-1,1] for c in [-1,1]])
                    corners=signs*np.asarray(box[3:6])/2
                    c,s=np.cos(box[6]),np.sin(box[6])
                    corners=corners@np.array([[c,s,0],[-s,c,0],[0,0,1]])+np.asarray(box[:3])
                    pts=(corners-np.asarray(cam['sensor2lidar_translation']))@np.asarray(cam['sensor2lidar_rotation'])
                    if np.any(pts[:,2]<=1):continue
                    uv=pts@np.asarray(cam['cam_intrinsic']).T;uv=uv[:,:2]/uv[:,2:]
                    x0,y0=np.floor(uv.min(0)).astype(int);x1,y1=np.ceil(uv.max(0)).astype(int)
                    if str(name)=='vehicle':y0=int(y0+.20*(y1-y0))
                    else:y1=int(y0+.40*(y1-y0))
                    x0=max(x0,0);x1=min(x1,rgb.shape[1]);y0=max(y0,0);y1=min(y1,rgb.shape[0])
                    if x1>x0 and y1>y0:
                        roi=rgb[y0:y1,x0:x1]
                        rgb[y0:y1,x0:x1]=cv2.resize(cv2.resize(roi,(max(1,(x1-x0)//35),max(1,(y1-y0)//35))), (x1-x0,y1-y0), interpolation=cv2.INTER_NEAREST)
                image=Image.fromarray(rgb)
                image.thumbnail((944, 580), Image.Resampling.LANCZOS)
                canvas = Image.new('RGB', (1600, 900), '#0e1726')
                draw = ImageDraw.Draw(canvas)
                draw.text((32, 24), args.model_label, font=font(32), fill='white')
                draw.text((32, 69), 'NAVSIM NAVTEST | Recorded-scene replay | 2 Hz | 1x speed', font=font(20), fill='#a9bbd0')
                draw.text((32, 120), 'FRONT CAMERA', font=font(20), fill='#56d5c4')
                canvas.paste(image, (32, 160+(580-image.height)//2))
                draw.text((1008, 120), 'BIRD\'S-EYE VIEW', font=font(20), fill='#56d5c4')
                location = frame['map_location']
                if location not in maps:
                    maps[location] = get_maps_api(str(args.maps), args.map_version, location)
                origin = np.asarray(frame['ego2global_translation'][:2])
                yaw = Quaternion(*frame['ego2global_rotation']).yaw_pitch_roll[0]
                rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                layers = [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION,
                          SemanticMapLayer.CROSSWALK, SemanticMapLayer.WALKWAYS]
                objects = maps[location].get_proximal_map_objects(Point2D(*origin), 70, layers)
                fig, ax = plt.subplots(figsize=(5.6, 5.8), dpi=100)
                fig.patch.set_facecolor('#edf2f6'); ax.set_facecolor('#edf2f6')
                for layer in layers:
                    for obj in objects[layer]:
                        xy = (np.asarray(obj.polygon.exterior.coords)[:, :2]-origin)@rotation
                        color = '#d5dde5' if layer in layers[:2] else '#b8d7cf'
                        ax.add_patch(Polygon(xy[:, ::-1], facecolor=color, edgecolor='#b4c0cc', linewidth=.35))
                for box, name in zip(frame['anns']['gt_boxes'], frame['anns']['gt_names']):
                    x,y,_,length,width,_,heading = box[:7]
                    if not -10 < x < 50 or abs(y)>32:
                        continue
                    corners = np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*np.array([length,width])/2
                    c,s=np.cos(heading),np.sin(heading)
                    corners=corners@np.array([[c,s],[-s,c]])+np.array([x,y])
                    color='#7a90ae' if str(name)=='vehicle' else '#ae7baf'
                    ax.add_patch(Polygon(corners[:,::-1], facecolor=color, edgecolor='#415671', linewidth=.6, zorder=4))
                # Rear-axle trajectory and vehicle footprint share the native ego frame.
                ax.add_patch(Polygon([[-1,-1],[1,-1],[1,4],[-1,4]],facecolor='#1b283d',zorder=6))
                xy=np.vstack(([0,0],trajectory[:,:2]))
                ax.plot(xy[:,1],xy[:,0],color='white',linewidth=5,zorder=7)
                ax.plot(xy[:,1],xy[:,0],color='#e66c2c',linewidth=3,zorder=8)
                ax.scatter(xy[1:,1],xy[1:,0],c=np.arange(8),cmap='autumn',s=27,edgecolors='#643f22',linewidths=.4,zorder=9)
                ax.set_xlim(29,-29);ax.set_ylim(-8,50);ax.set_aspect('equal')
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                fig.subplots_adjust(0,0,1,1);fig.canvas.draw()
                bev=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]);plt.close(fig)
                canvas.paste(bev,(1008,160))
                elapsed=(timestamp-int(frames[clip['start']]['timestamp']))/1e6
                turn='LEFT TURN' if np.mean(trajectory[-3:,1])>2 else ('RIGHT TURN' if np.mean(trajectory[-3:,1])<-2 else 'FORWARD')
                draw.text((32,755),f'CLIP {clip_index+1:02d}  /  {turn}  /  {elapsed:04.1f} s',font=font(25),fill='white')
                draw.text((32,800),'Orange: saved 4-second plan   |   BEV: map + annotated current actors',font=font(20),fill='#a9bbd0')
                draw.text((32,839),'No synthetic frames or plan interpolation. Not a closed-loop driving demonstration.',font=font(18),fill='#a9bbd0')
                draw.text((32,870),'Data: nuPlan / Motional + OpenScene | CC BY-NC-SA 4.0 + dataset terms | Camera privacy masking applied',font=font(14),fill='#7f94ad')
                writer.write(cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR))
                if index in [clip['start'], (clip['start']+clip['end'])//2, clip['end']]:
                    canvas.save(args.output/f'preview_{clip_index+1}_{index}.jpg',quality=92)
                records.append({'clip':clip_index+1,'frame_index':index,'timestamp':timestamp,
                                'prediction_sha256':hashlib.sha256(trajectory.tobytes()).hexdigest()})
                print('RENDERED',clip_index+1,index,flush=True)
    finally:
        writer.release()
    report={'model':args.model_label,'source_submission_sha256':hashlib.sha256(args.submission.read_bytes()).hexdigest(),
            'fps':2,'duration_seconds':len(records)/2,'frames':len(records),'clips':clips,
            'prediction_interpolation':False,'closed_loop':False,'camera_privacy_masking':'annotation-projected vehicle lower regions and pedestrian/bicycle upper regions','records':records}
    (args.output/'video_provenance.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
