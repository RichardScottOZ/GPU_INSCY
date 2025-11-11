#include "GPU_SCY_tree.cuh"
#include "../utils/TmpMalloc.cuh"
#include "../utils/util.cuh"

#include <math.h>
#ifndef M_PI_F
#define M_PI_F 3.141592654f
#endif

#define BLOCK_WIDTH 64
#define BLOCK_SIZE 512

#define PI 3.141592654f

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}










// OLD (delete or comment out)
/// NEW
#include <math.h>
#ifndef M_PI_F
#define M_PI_F 3.141592654f
#endif

// ===== BEGIN: prune helpers without libdevice special functions =====
// Uses only simple float math; avoids powf/tgammaf in device code.

#include <cuda_runtime.h>
#include <math_constants.h>  // CUDART_PI_F
#include "SCY_helpers.cuh"
#include "GPU_SCY_tree_kernels.cuh"

// Γ(k + 2) == (k + 1)! as float


// π^{k/2} without powf: π^m * sqrt(π) if k is odd


// c(k) = π^{k/2} / Γ(k+2)


// alpha(k) = 2 n eps^k c(k) / [ v^k (k + 2) ]


// expDen(k) = n c(k) eps^k / v^k



// ===== END: prune helpers without libdevice special functions =====




























































void GPU_SCY_tree::copy_to_device() {
    cudaMemcpy(d_parents, h_parents, sizeof(int) * this->number_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cells, h_cells, sizeof(int) * this->number_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, h_counts, sizeof(int) * this->number_of_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dim_start, h_dim_start, sizeof(int) * this->number_of_dims, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dims, h_dims, sizeof(int) * this->number_of_dims, cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, h_points, sizeof(int) * this->number_of_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_points_placement, h_points_placement, sizeof(int) * this->number_of_points,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_restricted_dims, h_restricted_dims, sizeof(int) * this->number_of_restricted_dims,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}


GPU_SCY_tree::GPU_SCY_tree(TmpMalloc *tmps, int number_of_nodes, int number_of_dims, int number_of_restricted_dims,
                           int number_of_points, int number_of_cells, float *mins, float *maxs, float v) {

    this->mins = mins;
    this->maxs = maxs;
    this->v = v;

    this->tmps = tmps;
    this->number_of_nodes = number_of_nodes;
    this->number_of_dims = number_of_dims;
    this->number_of_restricted_dims = number_of_restricted_dims;
    this->number_of_points = number_of_points;
    this->number_of_cells = number_of_cells;
    gpuErrchk(cudaPeekAtLastError());

    this->h_parents = new int[number_of_nodes];
    zero(this->h_parents, number_of_nodes);

    this->h_cells = new int[number_of_nodes];
    zero(this->h_cells, number_of_nodes);

    this->h_counts = new int[number_of_nodes];
    zero(this->h_counts, number_of_nodes);

    this->h_dim_start = new int[number_of_dims];
    zero(this->h_dim_start, number_of_dims);

    this->h_dims = new int[number_of_dims];
    zero(this->h_dims, number_of_dims);

    this->h_points = new int[number_of_points];
    zero(this->h_points, number_of_points);

    this->h_points_placement = new int[number_of_points];
    zero(this->h_points_placement, number_of_points);

    this->h_restricted_dims = new int[number_of_restricted_dims];
    zero(this->h_restricted_dims, number_of_restricted_dims);

    gpuErrchk(cudaPeekAtLastError());

    if (number_of_nodes > 0) {
        this->d_parents = tmps->malloc_nodes();
        gpuErrchk(cudaPeekAtLastError());

        this->d_cells = tmps->malloc_nodes();
        gpuErrchk(cudaPeekAtLastError());

        this->d_counts = tmps->malloc_nodes();
        gpuErrchk(cudaPeekAtLastError());
    }

    if (number_of_dims > 0) {
        this->d_dim_start = tmps->malloc_dims();
        this->d_dims = tmps->malloc_dims();
        gpuErrchk(cudaPeekAtLastError());
    }

    if (number_of_restricted_dims > 0) {
        this->d_restricted_dims = tmps->malloc_dims();
        gpuErrchk(cudaPeekAtLastError());
    }

    if (number_of_points > 0) {
        this->d_points = tmps->malloc_points();
        gpuErrchk(cudaPeekAtLastError());

        this->d_points_placement = tmps->malloc_points();
        gpuErrchk(cudaPeekAtLastError());
    }
}


GPU_SCY_tree::GPU_SCY_tree(int number_of_nodes, int number_of_dims, int number_of_restricted_dims, int number_of_points,
                           int number_of_cells, float *mins, float *maxs, float v) {

    this->mins = mins;
    this->maxs = maxs;
    this->v = v;

    this->number_of_nodes = number_of_nodes;
    this->number_of_dims = number_of_dims;
    this->number_of_restricted_dims = number_of_restricted_dims;
    this->number_of_points = number_of_points;
    this->number_of_cells = number_of_cells;

    this->h_parents = new int[number_of_nodes];
    zero(this->h_parents, number_of_nodes);

    this->h_cells = new int[number_of_nodes];
    zero(this->h_cells, number_of_nodes);

    this->h_counts = new int[number_of_nodes];
    zero(this->h_counts, number_of_nodes);

    this->h_dim_start = new int[number_of_dims];
    zero(this->h_dim_start, number_of_dims);

    this->h_dims = new int[number_of_dims];
    zero(this->h_dims, number_of_dims);

    this->h_points = new int[number_of_points];
    zero(this->h_points, number_of_points);

    this->h_points_placement = new int[number_of_points];
    zero(this->h_points_placement, number_of_points);

    this->h_restricted_dims = new int[number_of_restricted_dims];
    zero(this->h_restricted_dims, number_of_restricted_dims);

    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    if (number_of_nodes > 0) {
        cudaMalloc(&this->d_parents, number_of_nodes * sizeof(int));
        cudaMemset(this->d_parents, 0, number_of_nodes * sizeof(int));

        cudaMalloc(&this->d_cells, number_of_nodes * sizeof(int));
        cudaMemset(this->d_cells, 0, number_of_nodes * sizeof(int));

        cudaMalloc(&this->d_counts, number_of_nodes * sizeof(int));
        cudaMemset(this->d_counts, 0, number_of_nodes * sizeof(int));

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    if (number_of_dims > 0) {
        cudaMalloc(&this->d_dim_start, number_of_dims * sizeof(int));
        cudaMemset(this->d_dim_start, 0, number_of_dims * sizeof(int));

        cudaMalloc(&this->d_dims, number_of_dims * sizeof(int));
        cudaMemset(this->d_dims, 0, number_of_dims * sizeof(int));

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    if (number_of_restricted_dims > 0) {
        cudaMalloc(&this->d_restricted_dims, number_of_restricted_dims * sizeof(int));
        cudaMemset(this->d_restricted_dims, 0, number_of_restricted_dims * sizeof(int));

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    if (number_of_points > 0) {
        cudaMalloc(&this->d_points, number_of_points * sizeof(int));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemset(this->d_points, 0, number_of_points * sizeof(int));
        gpuErrchk(cudaPeekAtLastError());

        cudaMalloc(&this->d_points_placement, number_of_points * sizeof(int));
        gpuErrchk(cudaPeekAtLastError());
        cudaMemset(this->d_points_placement, 0, number_of_points * sizeof(int));

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }
}

vector <vector<GPU_SCY_tree *>>
GPU_SCY_tree::restrict_merge(TmpMalloc *tmps, int first_dim_no, int number_of_dims, int number_of_cells) {
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaPeekAtLastError());

    GPU_SCY_tree *scy_tree = this;

    tmps->reset_counters();


    int number_of_blocks;
    dim3 block(128);
    dim3 grid(number_of_dims, number_of_cells);

    int c = scy_tree->number_of_cells;
    int d = scy_tree->number_of_dims;

    int total_number_of_dim = first_dim_no + number_of_dims;
    int number_of_restrictions = number_of_dims * number_of_cells;

    vector <vector<GPU_SCY_tree *>> L(number_of_dims);

    vector <vector<GPU_SCY_tree *>> L_merged(number_of_dims);

    if (scy_tree->number_of_nodes * number_of_restrictions == 0)
        return L_merged;
    gpuErrchk(cudaPeekAtLastError());

    int *d_new_indecies = tmps->get_int_array(tmps->int_array_counter++, scy_tree->number_of_nodes *
                                                                         number_of_restrictions);
    gpuErrchk(cudaPeekAtLastError());
    int *d_new_counts = tmps->get_int_array(tmps->int_array_counter++,
                                            scy_tree->number_of_nodes * number_of_restrictions);
    gpuErrchk(cudaPeekAtLastError());
    int *d_is_included = tmps->get_int_array(tmps->int_array_counter++,
                                             scy_tree->number_of_nodes * number_of_restrictions);
    gpuErrchk(cudaPeekAtLastError());
    int *d_children_full = tmps->get_int_array(tmps->int_array_counter++,
                                               2 * scy_tree->number_of_nodes * number_of_restrictions *
                                               scy_tree->number_of_cells);
    gpuErrchk(cudaPeekAtLastError());

    int *d_parents_full = tmps->get_int_array(tmps->int_array_counter++,
                                              scy_tree->number_of_nodes * number_of_restrictions);
    gpuErrchk(cudaPeekAtLastError());

    cudaMemset(d_new_indecies, 0, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMemset(d_new_counts, 0, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMemset(d_is_included, 0, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMemset(d_parents_full, -1, scy_tree->number_of_nodes * number_of_restrictions * sizeof(int));
    cudaMemset(d_children_full, -1,
               2 * scy_tree->number_of_nodes * number_of_restrictions * scy_tree->number_of_cells * sizeof(int));
    for (int i = 0; i < number_of_dims; i++) {
        for (int cell_no = 0; cell_no < number_of_cells; cell_no++) {
            int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
            //old memset << < 1, 1 >> > (d_is_included + node_offset, 0, 1);
            set_int_at<<<1, 1>>>(d_is_included + node_offset, 0, 1);
        }
    }
    gpuErrchk(cudaPeekAtLastError());

    int *d_is_point_included = tmps->get_int_array(tmps->int_array_counter++, this->number_of_points *
                                                                              number_of_restrictions);
    gpuErrchk(cudaPeekAtLastError());
    int *d_point_new_indecies = tmps->get_int_array(tmps->int_array_counter++, this->number_of_points *
                                                                               number_of_restrictions);
    gpuErrchk(cudaPeekAtLastError());

    cudaMemset(d_is_point_included, 0, this->number_of_points * number_of_restrictions * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());

    int *d_is_s_connected = tmps->get_int_array(tmps->int_array_counter++,
                                                number_of_restrictions);
    cudaMemset(d_is_s_connected, 0, number_of_restrictions * sizeof(int));
    gpuErrchk(cudaPeekAtLastError());

    int *d_dim_i = tmps->get_int_array(tmps->int_array_counter++, number_of_dims);

    gpuErrchk(cudaPeekAtLastError());

    int *h_new_number_of_points = new int[number_of_restrictions];
    int *h_new_number_of_nodes = new int[number_of_restrictions];

    int *d_merge_map = tmps->get_int_array(tmps->int_array_counter++, number_of_restrictions);
    int *h_merge_map = new int[number_of_restrictions];

    int dim_no = first_dim_no;
    while (dim_no < total_number_of_dim) {
        int i = dim_no - first_dim_no;
        L[i] = vector<GPU_SCY_tree *>(number_of_cells);

        find_dim_i << < 1, 1 >> >
        (d_dim_i + i, scy_tree->d_dims, dim_no, scy_tree->number_of_dims);
        dim_no++;
    }

    if (number_of_dims > 0) {

        check_is_s_connected << < number_of_dims, block >> >
        (scy_tree->d_parents, scy_tree->d_cells, scy_tree->d_counts, scy_tree->d_dim_start,
                d_is_included, d_new_counts, d_is_s_connected, d_dim_i,
                scy_tree->number_of_dims, scy_tree->number_of_nodes,
                scy_tree->number_of_cells, scy_tree->number_of_points);
        gpuErrchk(cudaPeekAtLastError());

        compute_merge_map << < 1, number_of_dims >> >
        (d_is_s_connected, d_merge_map, scy_tree->number_of_cells);
        gpuErrchk(cudaPeekAtLastError());
        cudaMemcpy(h_merge_map, d_merge_map, number_of_restrictions * sizeof(int), cudaMemcpyDeviceToHost);
        gpuErrchk(cudaPeekAtLastError());

        restrict_merge_dim << < number_of_dims, block >> >
        (d_parents_full, scy_tree->d_parents, scy_tree->d_cells, scy_tree->d_counts, scy_tree->d_dim_start,
                d_is_included, d_new_counts, d_is_s_connected, d_dim_i, d_merge_map,
                scy_tree->number_of_dims, scy_tree->number_of_nodes,
                scy_tree->number_of_cells, scy_tree->number_of_points);
        gpuErrchk(cudaPeekAtLastError());

        restrict_dim_prop_up << < grid, block >> >
        (d_parents_full, d_children_full, scy_tree->d_parents, scy_tree->d_counts, scy_tree->d_cells, scy_tree->d_dim_start,
                d_is_included, d_new_counts, d_dim_i,
                scy_tree->number_of_dims, scy_tree->number_of_nodes,
                scy_tree->number_of_cells, scy_tree->number_of_points);

        gpuErrchk(cudaPeekAtLastError());

        restrict_merge_dim_prop_down_first << < grid, block >> >
        (d_parents_full, d_children_full, scy_tree->d_parents, scy_tree->d_counts, scy_tree->d_cells, scy_tree->d_dim_start,
                d_is_included, d_new_counts, d_dim_i, d_merge_map,
                scy_tree->number_of_dims, scy_tree->number_of_nodes,
                scy_tree->number_of_cells, scy_tree->number_of_points);

        gpuErrchk(cudaPeekAtLastError());

        restrict_dim_prop_down << < grid, block >> >
        (d_parents_full, d_children_full, scy_tree->d_parents, scy_tree->d_counts, scy_tree->d_cells, scy_tree->d_dim_start,
                d_is_included, d_new_counts, d_dim_i,
                scy_tree->number_of_dims, scy_tree->number_of_nodes,
                scy_tree->number_of_cells, scy_tree->number_of_points);

        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;

                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    // 2. do a scan to find the new indices for the nodes in the restricted tree
                    cudaDeviceSynchronize();
                    gpuErrchk(cudaPeekAtLastError());
                    inclusive_scan_nodes(d_is_included + node_offset, d_new_indecies + node_offset,
                                         scy_tree->number_of_nodes, tmps);

                    // 3. construct restricted tree
                    gpuErrchk(cudaPeekAtLastError());

                    int new_number_of_points = 0;
                    cudaMemcpy(&new_number_of_points, d_new_counts + node_offset, sizeof(int),
                               cudaMemcpyDeviceToHost);
                    gpuErrchk(cudaPeekAtLastError());

                    int new_number_of_nodes = 0;
                    cudaMemcpy(&new_number_of_nodes, d_new_indecies + node_offset + scy_tree->number_of_nodes - 1,
                               sizeof(int), cudaMemcpyDeviceToHost);
                    gpuErrchk(cudaPeekAtLastError());

                    cudaDeviceSynchronize();
                    gpuErrchk(cudaPeekAtLastError());

                    float ra = this->maxs[dim_no] - this->mins[dim_no];
                    GPU_SCY_tree *restricted_scy_tree = new GPU_SCY_tree(tmps, new_number_of_nodes,
                                                                         scy_tree->number_of_dims - 1,
                                                                         scy_tree->number_of_restricted_dims + 1,
                                                                         new_number_of_points,
                                                                         scy_tree->number_of_cells, mins, maxs,
                                                                         scy_tree->v * ra);
                    restricted_scy_tree->v_max = this->v_max;
                    gpuErrchk(cudaPeekAtLastError());

                    L[i][cell_no] = restricted_scy_tree;
                    L_merged[i].push_back(restricted_scy_tree);

                    restricted_scy_tree->is_s_connected = false;

                    cudaDeviceSynchronize();
                    gpuErrchk(cudaPeekAtLastError());
                }
                cell_no++;
            }
            dim_no++;
        }

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;
                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    GPU_SCY_tree *restricted_scy_tree = L[i][cell_no];

                    number_of_blocks = scy_tree->number_of_nodes / BLOCK_WIDTH;
                    if (scy_tree->number_of_nodes % BLOCK_WIDTH) number_of_blocks++;
                    restrict_move<<< number_of_blocks, BLOCK_WIDTH, 0 >>>
                            (d_parents_full + node_offset, scy_tree->d_cells, restricted_scy_tree->d_cells,
                             scy_tree->d_parents, restricted_scy_tree->d_parents, d_new_counts + node_offset,
                             restricted_scy_tree->d_counts, d_new_indecies + node_offset,
                             d_is_included + node_offset, scy_tree->number_of_nodes);
                }
                cell_no++;
            }
            dim_no++;
        }
        gpuErrchk(cudaPeekAtLastError());

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;
                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    GPU_SCY_tree *restricted_scy_tree = L[i][cell_no];

                    if (scy_tree->number_of_dims > 1) {

                        number_of_blocks = restricted_scy_tree->number_of_dims / BLOCK_WIDTH;
                        if (restricted_scy_tree->number_of_dims % BLOCK_WIDTH) number_of_blocks++;

                        restrict_update_dim << < number_of_blocks, BLOCK_WIDTH, 0 >> >
                        (scy_tree->d_dim_start, scy_tree->d_dims,
                                restricted_scy_tree->d_dim_start,
                                restricted_scy_tree->d_dims,
                                d_new_indecies +
                                node_offset,
                                d_dim_i +
                                i, restricted_scy_tree->number_of_dims);

                    }
                }
                cell_no++;
            }
            dim_no++;
        }
        gpuErrchk(cudaPeekAtLastError());

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;
                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    GPU_SCY_tree *restricted_scy_tree = L[i][cell_no];

                    number_of_blocks = restricted_scy_tree->number_of_restricted_dims / BLOCK_WIDTH;
                    if (restricted_scy_tree->number_of_restricted_dims % BLOCK_WIDTH) number_of_blocks++;
                    restrict_update_restricted_dim << < number_of_blocks, BLOCK_WIDTH, 0 >> >
                    (dim_no, scy_tree->d_restricted_dims, restricted_scy_tree->d_restricted_dims, scy_tree->number_of_restricted_dims);

                    for (int k = 0; k < scy_tree->number_of_restricted_dims; k++) {
                        restricted_scy_tree->h_restricted_dims[k] = scy_tree->h_restricted_dims[k];
                    }
                    restricted_scy_tree->h_restricted_dims[scy_tree->number_of_restricted_dims] = dim_no;
                }
                cell_no++;
            }
            dim_no++;
        }
        gpuErrchk(cudaPeekAtLastError());

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;

                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    GPU_SCY_tree *restricted_scy_tree = L[i][cell_no];

                    number_of_blocks = number_of_points / BLOCK_WIDTH;
                    if (number_of_points % BLOCK_WIDTH) number_of_blocks++;
                    restrict_merge_is_points_included
                    <<< number_of_blocks, BLOCK_WIDTH, 0 >>>
                            (d_parents_full + node_offset, scy_tree->d_points_placement, scy_tree->d_cells,
                             d_is_included + node_offset,
                             d_is_point_included + point_offset,
                             d_dim_i + i,
                             d_merge_map + i * number_of_cells,
                             scy_tree->number_of_dims, scy_tree->number_of_points, cell_no);

                }
                cell_no++;
            }
            dim_no++;
        }
        gpuErrchk(cudaPeekAtLastError());

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;

                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    GPU_SCY_tree *restricted_scy_tree = L[i][cell_no];

                    inclusive_scan_points(d_is_point_included + point_offset,
                                          d_point_new_indecies + point_offset,
                                          number_of_points, tmps);
                }
                cell_no++;
            }
            dim_no++;
        }
        gpuErrchk(cudaPeekAtLastError());

        dim_no = first_dim_no;
        while (dim_no < total_number_of_dim) {
            int i = dim_no - first_dim_no;
            int cell_no = 0;
            while (cell_no < number_of_cells) {
                int point_offset = i * number_of_cells * number_of_points + cell_no * number_of_points;
                int node_offset = i * number_of_cells * number_of_nodes + cell_no * number_of_nodes;
                int one_offset = i * number_of_cells + cell_no;
                if (cell_no == 0 || h_merge_map[one_offset - 1] != h_merge_map[one_offset]) {
                    GPU_SCY_tree *restricted_scy_tree = L[i][cell_no];

                    if (restricted_scy_tree->number_of_points > 0) {


                        move_points <<< number_of_blocks, BLOCK_WIDTH, 0 >>>
                                (d_parents_full + node_offset, d_children_full
                                                               + 2 * i * number_of_cells * number_of_cells *
                                                                 number_of_nodes
                                                               + 2 * cell_no * number_of_nodes * number_of_cells,
                                 scy_tree->d_parents, scy_tree->d_cells,
                                 scy_tree->d_points, scy_tree->d_points_placement, restricted_scy_tree->d_points,
                                 restricted_scy_tree->d_points_placement,
                                 d_point_new_indecies +
                                 point_offset,
                                 d_new_indecies + node_offset,
                                 d_is_point_included + point_offset,
                                 d_dim_i + i,
                                 number_of_points, scy_tree->number_of_dims, scy_tree->number_of_cells);
                        gpuErrchk(cudaPeekAtLastError());

                    }
                }
                cell_no++;
            }
            dim_no++;
        }
        gpuErrchk(cudaPeekAtLastError());
    }

    delete[] h_new_number_of_points;
    delete[] h_new_number_of_nodes;
    delete[] h_merge_map;

    return L_merged;
}

bool
GPU_SCY_tree::pruneRecursion(TmpMalloc *tmps, int min_size, float *d_X, int n, int d,
                             float neighborhood_size, float F,
                             int num_obj, int *d_neighborhoods, int *d_neighborhood_end,
                             bool rectangular) {


    if (this->number_of_points < min_size) {
        return false;
    }
    int blocks_points = this->number_of_points / 512;
    if (this->number_of_points % 512) blocks_points++;
    int blocks_nodes = this->number_of_nodes / 512;
    if (this->number_of_nodes % 512) blocks_nodes++;

    int *d_is_dense = tmps->malloc_points();
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    int *d_new_indices = tmps->malloc_points();
    gpuErrchk(cudaPeekAtLastError());
    cudaMemset(d_new_indices, 0, sizeof(int) * this->number_of_points);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    if (rectangular) {
        compute_is_weak_dense_rectangular_prune <<< blocks_points, min(512, this->number_of_points) >>>(d_is_dense,
                                                                                                        d_neighborhoods,
                                                                                                        d_neighborhood_end,
                                                                                                        this->d_points,
                                                                                                        this->number_of_points,
                                                                                                        this->d_restricted_dims,
                                                                                                        this->number_of_restricted_dims,
                                                                                                        d_X, n, d,
                                                                                                        F, num_obj,
                                                                                                        neighborhood_size,
                                                                                                        this->v_max);//todo fix volumn - not a problem since we run on standadized data
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    } else {
        compute_is_weak_dense_prune <<< blocks_points, min(512, this->number_of_points) >>>(d_is_dense, d_neighborhoods,
                                                                                            d_neighborhood_end,
                                                                                            this->d_points,
                                                                                            this->number_of_points,
                                                                                            this->d_restricted_dims,
                                                                                            this->number_of_restricted_dims,
                                                                                            d_X, n, d,
                                                                                            F, num_obj,
                                                                                            neighborhood_size, this->v_max);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }

    inclusive_scan_points(d_is_dense, d_new_indices, this->number_of_points, tmps);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    int new_number_of_points;
    cudaMemcpy(&new_number_of_points, d_new_indices + this->number_of_points - 1, sizeof(int), cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());

    if (new_number_of_points == 0) {
        tmps->free_points(d_is_dense);
        tmps->free_points(d_new_indices);
        return false;
    }

    int *d_new_points = tmps->malloc_points();
    gpuErrchk(cudaPeekAtLastError());

    int *d_new_point_placement = tmps->malloc_points();
    gpuErrchk(cudaPeekAtLastError());

    reset_counts_prune<<<blocks_nodes, min(512, this->number_of_nodes)>>>(this->d_counts, this->number_of_nodes);
    gpuErrchk(cudaPeekAtLastError());

    remove_pruned_points_prune <<< 1, min(512, this->number_of_points) >>>(d_is_dense, d_new_indices,
                                                                           d_new_points, d_new_point_placement,
                                                                           this->d_points, this->d_points_placement,
                                                                           this->number_of_points,
                                                                           this->d_counts, this->d_parents,
                                                                           this->number_of_nodes);
    gpuErrchk(cudaPeekAtLastError());


    tmps->free_points(this->d_points);
    tmps->free_points(this->d_points_placement);
    tmps->free_points(d_is_dense);
    tmps->free_points(d_new_indices);
    gpuErrchk(cudaPeekAtLastError());

    this->d_points = d_new_points;
    this->d_points_placement = d_new_point_placement;
    this->number_of_points = new_number_of_points;

    gpuErrchk(cudaPeekAtLastError());


    int *d_is_included = tmps->malloc_nodes();
    gpuErrchk(cudaPeekAtLastError());
    cudaMemset(d_is_included, 0, sizeof(int) * this->number_of_nodes);
    gpuErrchk(cudaPeekAtLastError());

    d_new_indices = tmps->malloc_nodes();
    gpuErrchk(cudaPeekAtLastError());
    cudaMemset(d_new_indices, 0, sizeof(int) * this->number_of_nodes);
    gpuErrchk(cudaPeekAtLastError());

    int *d_has_child = tmps->get_int_array(tmps->int_array_counter++, this->number_of_nodes * this->number_of_cells);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemset(d_has_child, 0, sizeof(int) * this->number_of_nodes * this->number_of_cells);
    gpuErrchk(cudaPeekAtLastError());

    compute_has_child_prune << < blocks_nodes, min(512, this->number_of_nodes) >> > (d_has_child,
            this->d_parents, this->d_cells, this->d_counts, this->number_of_nodes, this->number_of_cells);

    gpuErrchk(cudaPeekAtLastError());

    compute_is_included_prune << < blocks_nodes, min(512, this->number_of_nodes) >> > (d_is_included, d_has_child,
            this->d_parents, this->d_cells, this->d_counts, this->number_of_nodes, this->number_of_cells);

    gpuErrchk(cudaPeekAtLastError());

    inclusive_scan_nodes(d_is_included, d_new_indices, this->number_of_nodes, tmps);

    gpuErrchk(cudaPeekAtLastError());

    int new_number_of_nodes;
    cudaMemcpy(&new_number_of_nodes, d_new_indices + this->number_of_nodes - 1, sizeof(int), cudaMemcpyDeviceToHost);

    gpuErrchk(cudaPeekAtLastError());
    if (new_number_of_nodes <= 0) {
        tmps->free_nodes(d_is_included);
        tmps->free_nodes(d_new_indices);
        return false;
    }

    int *d_new_parents = tmps->malloc_nodes();
    int *d_new_cells = tmps->malloc_nodes();
    int *d_new_counts = tmps->malloc_nodes();
    gpuErrchk(cudaPeekAtLastError());

    blocks_points = this->number_of_points / 512;
    if (this->number_of_points % 512) blocks_points++;
    update_point_placement << < blocks_points, min(512, this->number_of_points) >> >
    (d_new_indices, this->d_points_placement, this->number_of_points);

    gpuErrchk(cudaPeekAtLastError());

    remove_nodes << < blocks_nodes, min(512, this->number_of_nodes) >> >
    (d_new_indices, d_is_included, d_new_parents, d_new_cells, d_new_counts,
            this->d_parents, this->d_cells, this->d_counts, this->number_of_nodes);

    gpuErrchk(cudaPeekAtLastError());
    tmps->free_nodes(this->d_parents);
    tmps->free_nodes(this->d_cells);
    tmps->free_nodes(this->d_counts);
    gpuErrchk(cudaPeekAtLastError());


    this->d_parents = d_new_parents;
    this->d_cells = d_new_cells;
    this->d_counts = d_new_counts;
    this->number_of_nodes = new_number_of_nodes;

    if (this->number_of_dims > 0) {

        update_dim_start << < 1, min(512, this->number_of_dims) >> >
        (d_new_indices, this->d_dim_start, this->number_of_dims);

        gpuErrchk(cudaPeekAtLastError());
    }

    tmps->free_nodes(d_is_included);
    tmps->free_nodes(d_new_indices);

    return this->number_of_points >= min_size;
}


bool GPU_SCY_tree::pruneRedundancy(float r, map<vector<int>, int *, vec_cmp> result, int n, TmpMalloc *tmps) {

    tmps->reset_counters();

    int number_of_blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE) number_of_blocks++;
    int number_of_threads = min(n, BLOCK_SIZE);

    int max_min_size = 0;

    vector<int> subspace(this->h_restricted_dims, this->h_restricted_dims +
                                                  this->number_of_restricted_dims);
    vector<int> max_min_subspace;

    int *d_clustering_H;

    int *d_sizes_H = tmps->get_int_array(tmps->int_array_counter++, n);

    int *d_cluster_to_use = tmps->get_int_array(tmps->int_array_counter++, n);
    int *d_min_size = tmps->malloc_one();

    for (std::pair<vector<int>, int *> subspace_clustering : result) {

        // find sizes of clusters
        vector<int> subspace_mark = subspace_clustering.first;

        if (subspace_of(subspace, subspace_mark)) {

            d_clustering_H = subspace_clustering.second;
            cudaMemset(d_sizes_H, 0, n * sizeof(int));
            cudaMemset(d_cluster_to_use, 0, n * sizeof(int));
            prune_count_kernel << < 1, number_of_threads >> > (d_sizes_H, d_clustering_H, n);

            prune_to_use << < number_of_blocks, number_of_threads >> >
            (d_cluster_to_use, d_clustering_H, d_points, this->number_of_points);

            // find the minimum size for each subspace
            cudaMemset(d_min_size, -1, sizeof(int));
            int min_size;
            prune_min_cluster << < 1, number_of_threads >> >
            (d_min_size, d_cluster_to_use, d_sizes_H, d_clustering_H, n);
            cudaMemcpy(&min_size, d_min_size, sizeof(int), cudaMemcpyDeviceToHost);

            // find the maximum minimum size for each subspace
            if (min_size > max_min_size) {
                max_min_size = min_size;
                max_min_subspace = subspace_mark;
            }
        }
    }

    tmps->free_one(d_min_size);

    if (max_min_size == 0) {
        return true;
    }

    return this->number_of_points * r > max_min_size * 1.;
}


GPU_SCY_tree::~GPU_SCY_tree() {
    if (!freed_partial) {
        if (number_of_nodes > 0) {
            if (tmps == nullptr) {
                cudaFree(d_parents);
                cudaFree(d_cells);
                cudaFree(d_counts);
            } else {
                tmps->free_nodes(d_parents);
                tmps->free_nodes(d_cells);
                tmps->free_nodes(d_counts);
            }
            delete[] h_parents;
            delete[] h_cells;
            delete[] h_counts;
        }
        if (number_of_dims > 0) {
            if (tmps == nullptr) {
                cudaFree(d_dim_start);
                cudaFree(d_dims);
            } else {
                tmps->free_dims(d_dim_start);
                tmps->free_dims(d_dims);
            }
            delete[] h_dim_start;
            delete[] h_dims;
        }
        if (number_of_points > 0) {
            if (tmps == nullptr) {
                cudaFree(d_points_placement);
            } else {
                tmps->free_points(d_points_placement);
            }
            delete[] h_points_placement;
        }
    }
    if (number_of_restricted_dims > 0) {
        if (tmps == nullptr) {
            cudaFree(d_restricted_dims);
        } else {
            tmps->free_dims(d_restricted_dims);
        }
        delete[] h_restricted_dims;
    }
    if (number_of_points > 0) {
        if (tmps == nullptr) {
            cudaFree(d_points);
        } else {
            tmps->free_points(d_points);
        }
        delete[] h_points;
    }
}
