#pragma once
// Prototypes for kernels implemented in GPU_SCY_tree_kernels.cu
extern "C" {
__global__ void set_int_at(int *a, int i, int val);
__global__ void find_dim_i(int *d_dim_i, int *d_dims, int dim_no, int d);
__global__ void check_is_s_connected(int *d_parents, int *d_cells, int *d_counts, int *d_dim_start, int *d_is_included_full, int *d_new_counts_full, int *d_is_s_connected_full, int *d_dim_i_full, int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points);
__global__ void compute_merge_map(int *d_is_s_connected_full, int *d_merge_map_full, int number_of_cells);
__global__ void restrict_merge_dim(int *d_new_parents_full, int *d_parents, int *d_cells, int *d_counts, int *d_dim_start, int *d_is_included_full, int *d_new_counts_full, int *d_is_s_connected_full, int *d_dim_i_full, int *d_merge_map_full, int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points);
__global__ void restrict_dim_prop_up(int *d_new_parents_full, int *d_children_full, int *d_parents, int *d_counts, int *d_cells, int *d_dim_start, int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full, int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points);
__global__ void restrict_merge_dim_prop_down_first(int *d_new_parents_full, int *d_children_full, int *d_parents, int *d_counts, int *d_cells, int *d_dim_start, int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full, int *d_merge_map_full, int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points);
__global__ void restrict_dim_prop_down(int *d_new_parents_full, int *d_children_full, int *d_parents, int *d_counts, int *d_cells, int *d_dim_start, int *d_is_included_full, int *d_new_counts_full, int *d_dim_i_full, int number_of_dims, int number_of_nodes, int number_of_cells, int number_of_points);
__global__ void restrict_move(int *d_new_parents, int *d_cells_1, int *d_cells_2, int *d_parents_1, int *d_parents_2, int *d_new_counts, int *d_counts_2, int *d_new_indecies, int *d_is_included, int n);
__global__ void restrict_update_dim(int *dim_start_1, int *dims_1, int *dim_start_2, int *dims_2, int *new_indecies, int *d_dim_i, int d_2);
__global__ void restrict_update_restricted_dim(int restrict_dim, int *d_restricted_dims_1, int *d_restricted_dims_2, int number_of_restricted_dims_1);
__global__ void restrict_merge_is_points_included(int *d_new_parents, int *d_points_placement, int *d_cells, int *d_is_included, int *d_is_point_included, int *d_dim_i, int *d_merge_map, int number_of_dims, int number_of_points, int c_i);
__global__ void move_points(int *d_new_parents, int *d_children, int *d_parents, int *d_cells, int *d_points_1, int *d_points_placement_1, int *d_points_2, int *d_points_placement_2, int *d_point_new_indecies, int *d_new_indecies, int *d_is_point_included, int *d_dim_i, int number_of_points, int number_of_dims, int number_of_cells);
__global__ void compute_is_weak_dense_prune(int *d_is_dense, int *d_neighborhoods, int *d_neighborhood_end, int *d_points, int number_of_points, int *subspace, int subspace_size, float *X, int n, int d, float F, int num_obj, float neighborhood_size, float v);
__global__ void compute_is_weak_dense_rectangular_prune(int *d_is_dense, int *d_neighborhoods, int *d_neighborhood_end, int *d_points, int number_of_points, int *subspace, int subspace_size, float *X, int n, int d, float F, int num_obj, float neighborhood_size, float v);
__global__ void reset_counts_prune(int *d_counts, int number_of_nodes);
__global__ void remove_pruned_points_prune(int *d_is_dense, int *d_new_indices, int *d_new_points, int *d_new_point_placement, int *d_points, int *d_point_placement, int number_of_points, int *d_counts, int *d_parents, int number_of_nodes);
__global__ void compute_has_child_prune(int *d_has_child, int *d_parents, int *d_cells, int *d_counts, int number_of_nodes, int number_of_cells);
__global__ void compute_is_included_prune(int *d_is_included, int *d_has_child, int *d_parents, int *d_cells, int *d_counts, int number_of_nodes, int number_of_cells);
__global__ void update_point_placement(int *d_new_indices, int *d_points_placement, int number_of_points);
__global__ void remove_nodes(int *d_new_indices, int *d_is_included, int *d_new_parents, int *d_new_cells, int *d_new_counts, int *d_parents, int *d_cells, int *d_counts, int number_of_nodes);
__global__ void update_dim_start(int *d_new_indices, int *d_dim_start, int number_of_dims);
__global__ void prune_count_kernel(int *d_sizes, int *d_clustering, int n);
__global__ void prune_to_use(int *d_cluster_to_use, int *d_clustering_H, int *d_points, int number_of_points);
__global__ void prune_min_cluster(int *d_min_size, int *d_cluster_to_use, int *d_sizes, int *d_clustering, int n);
} // extern "C"

