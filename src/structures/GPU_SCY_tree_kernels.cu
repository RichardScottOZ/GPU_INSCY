
#include <cuda_runtime.h>
#include <math_constants.h>
#include "SCY_helpers.cuh"

// Keep block sizes consistent with original TU (if referenced)
#ifndef BLOCK_WIDTH
#define BLOCK_WIDTH 64
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

// Forward declarations are in GPU_SCY_tree_kernels.cuh (included by host TU)

__global__
void set_int_at(int *a, int i, int val) {
    a[i] = val;
}


__global__
void find_dim_i(int *d_dim_i, int *d_dims, int dim_no, int d) {
    for (int i = 0; i < d; i++) {
        if (d_dims[i] == dim_no) {
            d_dim_i[0] = i;
        }
    }
}


__global__
void check_is_s_connected(int *d_parents, int *d_cells, int *d_counts, int *d_dim_start,
                          int *d_is_included_full, int *d_new_counts_full, int *d_is_s_connected_full,
                          int *d_dim_i_full,
                          int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points) {

    int i = blockIdx.x;

    int dim_i = d_dim_i_full[i];
    int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i, number_of_dims, number_of_nodes);
    int lvl_start = d_dim_start[dim_i];

    for (int j = threadIdx.x; j < lvl_size; j += blockDim.x) {
        int cell_no = d_cells[lvl_start + j];

        int one_offset = i * number_of_cells + cell_no;

        if (d_counts[lvl_start + j] < 0 &&
            (d_parents[lvl_start + j] == 0 || d_counts[d_parents[lvl_start + j]] >= 0)) {
            d_is_s_connected_full[one_offset] = 1;
        }
    }
}


__global__
void compute_merge_map(int *d_is_s_connected_full, int *d_merge_map_full, int number_of_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int one_offset = i * number_of_cells;

    int *d_is_s_connected = d_is_s_connected_full + one_offset;
    int *d_merge_map = d_merge_map_full + one_offset;

    int prev_s_connected = false;
    int prev_cell_no = 0;
    for (int cell_no = 0; cell_no < number_of_cells; cell_no++) {
        if (prev_s_connected) {
            d_merge_map[cell_no] = prev_cell_no;
        } else {
            d_merge_map[cell_no] = cell_no;
        }

        prev_s_connected = d_is_s_connected[cell_no];
        prev_cell_no = d_merge_map[cell_no];
    }
}


__global__
void restrict_merge_dim(int *d_new_parents_full, int *d_parents, int *d_cells, int *d_counts, int *d_dim_start,
                        int *d_is_included_full, int *d_new_counts_full, int *d_is_s_connected_full,
                        int *d_dim_i_full, int *d_merge_map_full,
                        int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points) {

    int i = blockIdx.x;

    int *d_merge_map = d_merge_map_full + i * number_of_cells;

    int dim_i = d_dim_i_full[i];
    int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i, number_of_dims, number_of_nodes);
    int lvl_start = d_dim_start[dim_i];

    for (int j = threadIdx.x; j < lvl_size; j += blockDim.x) {

        int cell_no = d_merge_map[d_cells[lvl_start + j]];
        int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
        int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
        int one_offset = i * number_of_cells + cell_no;

        int *d_is_included = d_is_included_full + node_offset;
        int *d_new_counts = d_new_counts_full + node_offset;
        int *d_new_parents = d_new_parents_full + node_offset;
        int *d_is_s_connected = d_is_s_connected_full + one_offset;

        int count = d_counts[lvl_start + j] > 0 ? d_counts[lvl_start + j] : 0;
        d_is_included[d_parents[lvl_start + j]] = 1;
        d_new_parents[d_parents[lvl_start + j]] = d_parents[d_parents[lvl_start + j]];
        atomicAdd(&d_new_counts[d_parents[lvl_start + j]], count);
    }
}


__global__
void
restrict_dim_prop_up(int *d_new_parents_full, int *d_children_full, int *d_parents, int *d_counts, int *d_cells,
                     int *d_dim_start,
                     int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                     int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points) {

    int i = blockIdx.x;
    int cell_no = blockIdx.y;

    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;
    int *d_new_parents = d_new_parents_full + node_offset;
    int *d_children = d_children_full
                      + 2 * i * number_of_cells * number_of_cells * number_of_nodes
                      + 2 * cell_no * number_of_nodes * number_of_cells;

    int dim_i = d_dim_i_full[i];

    d_new_parents[0] = 0;

    for (int d_j = dim_i - 1; d_j >= 0; d_j--) {

        int lvl_size = get_lvl_size_gpu(d_dim_start, d_j, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[d_j];

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            atomicMax(&d_is_included[d_parents[n_i]], d_is_included[n_i]);
            atomicAdd(&d_new_counts[d_parents[n_i]],
                      d_new_counts[n_i] > 0 ? d_new_counts[n_i] : 0);
            if (d_counts[n_i] < 0) {
                d_new_counts[n_i] = -1;
            }

            int s_connection = d_counts[n_i] >= 0 ? 0 : 1;
            if (d_is_included[n_i]) {
                d_new_parents[d_parents[n_i]] = d_parents[d_parents[n_i]];
                int cell = d_cells[d_parents[n_i]] >= 0 ? d_cells[d_parents[n_i]] : 0;
                d_children[d_parents[d_parents[n_i]] * number_of_cells * 2 + 2 * cell +
                           s_connection] = n_i;
            }
        }
        __syncthreads();
    }
}


__global__
void
restrict_merge_dim_prop_down_first(int *d_new_parents_full, int *d_children_full, int *d_parents, int *d_counts,
                                   int *d_cells,
                                   int *d_dim_start,
                                   int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                                   int *d_merge_map_full,
                                   int number_of_dims, int number_of_nodes, int number_of_cells,
                                   int number_of_points) {
    int i = blockIdx.x;
    int cell_no = blockIdx.y;


    int *d_merge_map = d_merge_map_full + i * number_of_cells;

    if (cell_no > 0 && d_merge_map[cell_no] == d_merge_map[cell_no - 1]) {
        return;
    }

    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;
    int *d_new_parents = d_new_parents_full + node_offset;
    int *d_children = d_children_full
                      + 2 * i * number_of_cells * number_of_cells * number_of_nodes
                      + 2 * cell_no * number_of_nodes * number_of_cells;

    int dim_i = d_dim_i_full[i];


    if (dim_i + 1 < number_of_dims) {
        int lvl_size = get_lvl_size_gpu(d_dim_start, dim_i + 1, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[dim_i + 1];

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int new_parent = d_parents[d_parents[n_i]];
            int s_connection = d_counts[n_i] >= 0 ? 0 : 1;

            int is_cell_no = ((d_merge_map[d_cells[d_parents[n_i]]] == cell_no) ? 1 : 0);
            if (is_cell_no && !(d_counts[d_parents[n_i]] < 0 && d_counts[d_parents[d_parents[n_i]]] >= 0)) {
                atomicMax(&d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i] + s_connection], n_i);
                d_new_parents[n_i] = new_parent;
            }
        }

        __syncthreads();

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int new_parent = d_new_parents[n_i];
            if (new_parent >= 0) {
                int s_connection = d_counts[n_i] >= 0 ? 0 : 1;
                int n_new = d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i] + s_connection];

                if (n_i == n_new) {
                    atomicMax(&d_is_included[n_new], 1);
                }

                if (d_counts[n_i] >= 0) {
                    atomicAdd(&d_new_counts[n_new], d_counts[n_i]);
                } else {
                    d_new_counts[n_new] = -1;
                }
            }
        }
    }
}


__global__
void restrict_dim_prop_down(int *d_new_parents_full, int *d_children_full,
                            int *d_parents, int *d_counts, int *d_cells,
                            int *d_dim_start,
                            int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full,
                            int number_of_dims, int number_of_nodes, int number_of_cells,
                            int number_of_points) {
    int i = blockIdx.x;
    int cell_no = blockIdx.y;

    int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;

    int *d_is_included = d_is_included_full + node_offset;
    int *d_new_counts = d_new_counts_full + node_offset;
    int *d_new_parents = d_new_parents_full + node_offset;
    int *d_children = d_children_full
                      + 2 * i * number_of_cells * number_of_cells * number_of_nodes
                      + 2 * cell_no * number_of_nodes * number_of_cells;


    int dim_i = d_dim_i_full[i];


    for (int d_j = dim_i + 2; d_j < number_of_dims; d_j++) {
        int lvl_size = get_lvl_size_gpu(d_dim_start, d_j, number_of_dims, number_of_nodes);
        int lvl_start = d_dim_start[d_j];

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int s_connection = d_counts[n_i] >= 0 ? 0 : 1;
            int old_parent = d_parents[n_i];
            int parent_s_connection = d_counts[old_parent] >= 0 ? 0 : 1;
            int new_parent_parent = d_new_parents[old_parent];
            if (new_parent_parent >= 0) {
                int new_parent = d_children[new_parent_parent * number_of_cells * 2 +
                                            2 * d_cells[old_parent] + parent_s_connection];

                if (new_parent >= 0) {
                    d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i] + s_connection] = n_i;
                    d_new_parents[n_i] = new_parent;
                }
            }
        }

        __syncthreads();

        for (int i = threadIdx.x; i < lvl_size; i += blockDim.x) {
            int n_i = lvl_start + i;
            int new_parent = d_new_parents[n_i];
            int s_connection = d_counts[n_i] >= 0 ? 0 : 1;
            if (new_parent >= 0) {
                int n_new = d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i] + s_connection];
                if (n_i == n_new) {
                    atomicMax(&d_is_included[n_new], d_is_included[new_parent]);
                }

                if (d_counts[n_i] >= 0) {
                    atomicAdd(&d_new_counts[n_new], d_counts[n_i]);
                } else {
                    d_new_counts[n_new] = -1;
                }
            }
        }
        __syncthreads();
    }
}


__global__
void
restrict_move(int *d_new_parents, int *d_cells_1, int *d_cells_2, int *d_parents_1, int *d_parents_2,
              int *d_new_counts, int *d_counts_2,
              int *d_new_indecies, int *d_is_included, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d_is_included[i]) {
        int new_idx = d_new_indecies[i] - 1;
        d_cells_2[new_idx] = d_cells_1[i];
        int new_parent = d_new_parents[i];
        d_parents_2[new_idx] = d_new_indecies[new_parent] - 1;
        d_counts_2[new_idx] = d_new_counts[i];
    }
}


__global__
void restrict_update_dim(int *dim_start_1, int *dims_1, int *dim_start_2, int *dims_2, int *new_indecies,
                         int *d_dim_i,
                         int d_2) {
    int d_i_start = d_dim_i[0];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = j + (d_i_start <= j ? 1 : 0);
    if (j < d_2) {
        int idx = dim_start_1[i] - 1;
        dim_start_2[j] = idx >= 0 ? new_indecies[idx] : 0;
        dims_2[j] = dims_1[i];
    }
}


__global__
void
restrict_update_restricted_dim(int restrict_dim, int *d_restricted_dims_1, int *d_restricted_dims_2,
                               int number_of_restricted_dims_1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_restricted_dims_1)
        d_restricted_dims_2[i] = d_restricted_dims_1[i];
    if (i == number_of_restricted_dims_1)
        d_restricted_dims_2[i] = restrict_dim;
}


__global__
void
restrict_merge_is_points_included(int *d_new_parents, int *d_points_placement, int *d_cells, int *d_is_included,
                                  int *d_is_point_included, int *d_dim_i, int *d_merge_map,
                                  int number_of_dims, int number_of_points, int c_i) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int dim_i = d_dim_i[0];
    bool restricted_dim_is_leaf = (dim_i == number_of_dims - 1);

    if (i >= number_of_points) return;

    int is_included = 0;
    int new_parent = d_new_parents[d_points_placement[i]];
    if (new_parent >= 0)
        is_included = 1;

    if (restricted_dim_is_leaf && d_merge_map[d_cells[d_points_placement[i]]] == c_i) {
        is_included = 1;
    }

    d_is_point_included[i] = is_included;
}


__global__
void
move_points(int *d_new_parents, int *d_children,
            int *d_parents, int *d_cells, int *d_points_1, int *d_points_placement_1,
            int *d_points_2, int *d_points_placement_2,
            int *d_point_new_indecies, int *d_new_indecies,
            int *d_is_point_included, int *d_dim_i,
            int number_of_points, int number_of_dims, int number_of_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int dim_i = d_dim_i[0];
    bool restricted_dim_is_leaf = (dim_i == number_of_dims - 1);

    if (i >= number_of_points) return;

    if (d_is_point_included[i]) {
        int new_parent = d_new_parents[d_points_placement_1[i]];
        int old_parent = d_parents[d_points_placement_1[i]];
        d_points_2[d_point_new_indecies[i] - 1] = d_points_1[i];
        if (restricted_dim_is_leaf) {
            d_points_placement_2[d_point_new_indecies[i] - 1] =
                    d_new_indecies[old_parent] - 1;
        } else {
            int n_i = d_points_placement_1[i];
            int n_new = d_children[new_parent * number_of_cells * 2 + 2 * d_cells[n_i]];
            if (n_new < 0) {
            }
            d_points_placement_2[d_point_new_indecies[i] - 1] = d_new_indecies[n_new] - 1;
        }
    }
}


__global__
void compute_is_weak_dense_prune(int *d_is_dense, int *d_neighborhoods, int *d_neighborhood_end,
                                 int *d_points, int number_of_points,
                                 int *subspace, int subspace_size,
                                 float *X, int n, int d, float F, int num_obj,
                                 float neighborhood_size, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {

        int p_id = d_points[i];

        float p = 0;
        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        for (int j = offset; j < d_neighborhood_end[p_id]; j++) {
            int q_id = d_neighborhoods[j];
            if (q_id >= 0) {
                float distance = dist_prune_gpu(p_id, q_id, X, d, subspace, subspace_size) / neighborhood_size;
                float sq = distance * distance;
                p += (1. - sq);
            }
        }
        float a = alpha_prune_gpu(d, neighborhood_size, n, v);
        float w = omega_prune_gpu(d);
        d_is_dense[i] = p >= fmaxf(F * a, num_obj * w) ? 1 : 0;
    }
}


__global__
void compute_is_weak_dense_rectangular_prune(int *d_is_dense, int *d_neighborhoods, int *d_neighborhood_end,
                                             int *d_points, int number_of_points,
                                             int *subspace, int subspace_size,
                                             float *X, int n, int d, float F, int num_obj,
                                             float neighborhood_size, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {

        int p_id = d_points[i];
        int offset = p_id > 0 ? d_neighborhood_end[p_id - 1] : 0;
        int neighbor_count = d_neighborhood_end[p_id] - offset;
        float a = expDen_prune_gpu(d, neighborhood_size, n, v);
        d_is_dense[i] = neighbor_count >= fmaxf(F * a, (float) num_obj);
    }
}


__global__
void reset_counts_prune(int *d_counts, int number_of_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_nodes) {
        if (d_counts[i] > 0) {
            d_counts[i] = 0;
        }
    }
}


__global__
void remove_pruned_points_prune(int *d_is_dense, int *d_new_indices,
                                int *d_new_points, int *d_new_point_placement,
                                int *d_points, int *d_point_placement, int number_of_points,
                                int *d_counts, int *d_parents, int number_of_nodes) {
    for (int i = threadIdx.x; i < number_of_points; i += blockDim.x) {
        if (d_is_dense[i]) {
            int new_i = d_new_indices[i] - 1;
            d_new_points[new_i] = d_points[i];
            d_new_point_placement[new_i] = d_point_placement[i];
            int node = d_point_placement[i];
            atomicAdd(&d_counts[node], 1);
            int count = 0;
            while (d_parents[node] != node) {
                if (node <= d_parents[node]) {
                    break;
                }
                count++;
                node = d_parents[node];
                atomicAdd(&d_counts[node], 1);
            }
        }
    }
}


__global__
void compute_has_child_prune(int *d_has_child, int *d_parents, int *d_cells, int *d_counts, int number_of_nodes,
                             int number_of_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_nodes) {
        if (d_counts[i] > 0) {
            int cell = d_cells[i];
            int parent = d_parents[i];
            if (parent != i) {
                d_has_child[parent * number_of_cells + cell] = 1;
            }
        }
    }
}


__global__
void compute_is_included_prune(int *d_is_included, int *d_has_child,
                               int *d_parents, int *d_cells, int *d_counts, int number_of_nodes, int number_of_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_nodes) {
        int cell = d_cells[i];
        int parent = d_parents[i];
        if (parent == i || d_has_child[parent * number_of_cells + cell]) {
            d_is_included[i] = 1;
        }
    }
}


__global__
void update_point_placement(int *d_new_indices, int *d_points_placement, int number_of_points) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_points) {
        int placement = d_points_placement[i];
        d_points_placement[i] = d_new_indices[placement] - 1;
    }
}


__global__
void remove_nodes(int *d_new_indices, int *d_is_included, int *d_new_parents, int *d_new_cells, int *d_new_counts,
                  int *d_parents, int *d_cells, int *d_counts, int number_of_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_nodes) {
        if (d_is_included[i]) {
            int i_new = d_new_indices[i] - 1;
            int parent = d_parents[i];
            d_new_parents[i_new] = d_new_indices[parent] - 1;
            d_new_cells[i_new] = d_cells[i];
            d_new_counts[i_new] = d_counts[i];
        }
    }
}


__global__
void update_dim_start(int *d_new_indices, int *d_dim_start, int number_of_dims) {
    for (int i = threadIdx.x; i < number_of_dims; i += blockDim.x) {
        int idx = d_dim_start[i] - 1;
        d_dim_start[i] = idx >= 0 ? d_new_indices[idx] : 0;
    }
}


__global__
void prune_count_kernel(int *d_sizes, int *d_clustering, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int cluster = d_clustering[i];
        if (cluster >= 0) {
            atomicAdd(&d_sizes[cluster], 1);
        }
    }
}


__global__
void prune_to_use(int *d_cluster_to_use, int *d_clustering_H, int *d_points, int number_of_points) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < number_of_points; i += blockDim.x * gridDim.x) {
        int p_id = d_points[i];
        int cluster_id = d_clustering_H[p_id];
        if (cluster_id >= 0) {
            d_cluster_to_use[cluster_id] = 1;
        }
    }
}


__global__
void prune_min_cluster(int *d_min_size, int *d_cluster_to_use, int *d_sizes, int *d_clustering, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int size = d_sizes[i];
        if (d_cluster_to_use[i]) {
            atomicCAS(&d_min_size[0], -1, size);
            atomicMin(&d_min_size[0], size);
        }
    }
}
