#include <stdint.h>
#include <math.h>

/* Preserve sklearn HGB's serial tree-summation order and double thresholds.
 * Numeric splits only; callers reject categorical trees. No fast-math. */
void hist_predict(const double *x, int64_t n, int64_t width,
                  const int64_t *roots, int64_t trees,
                  const int64_t *left, const int64_t *right,
                  const int64_t *feature, const double *threshold,
                  const double *value, const uint8_t *leaf,
                  const uint8_t *missing_left, double baseline, double *out) {
    for (int64_t row = 0; row < n; row++) {
        double sum = baseline;
        for (int64_t tree = 0; tree < trees; tree++) {
            int64_t node = roots[tree];
            while (!leaf[node]) {
                double v = x[row * width + feature[node]];
                node = isnan(v) ? (missing_left[node] ? left[node] : right[node])
                                : (v <= threshold[node] ? left[node] : right[node]);
            }
            sum += value[node];
        }
        out[row] = sum;
    }
}

typedef struct {
    uint32_t left, right;
    int32_t feature;
    uint8_t leaf, missing_left;
    uint16_t padding;
    double threshold, value;
} PackedNode;

/* Tree-major traversal reuses each tree's top nodes across all candidates.
 * Per-output accumulation order is still exactly tree 0, 1, ..., T-1. */
void hist_predict_packed(const double *x, int64_t n, int64_t width,
                         const int64_t *roots, int64_t trees,
                         const PackedNode *nodes, double baseline, double *out) {
    for (int64_t row = 0; row < n; row++) out[row] = baseline;
    for (int64_t tree = 0; tree < trees; tree++) {
        for (int64_t row = 0; row < n; row++) {
            const PackedNode *node = nodes + roots[tree];
            while (!node->leaf) {
                double v = x[row * width + node->feature];
                uint32_t next = isnan(v) ? (node->missing_left ? node->left : node->right)
                                          : (v <= node->threshold ? node->left : node->right);
                node = nodes + next;
            }
            out[row] += node->value;
        }
    }
}
