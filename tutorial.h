/*
 * Copyright (C) 2019  Jimmy Aguilar Mena
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef MATVEC_H
#define MATVEC_H

#include <nanos6.h>
#include <nanos6/debug.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <assert.h>


static inline void init(double *ret, const size_t rows, size_t cols, size_t ts)
{
	const size_t numNodes = nanos6_get_num_cluster_nodes();
	assert(rows >= ts);              // at least 1 portion per task
	assert(rows / ts >= numNodes);   // at least 1 task / node.
	assert(rows % ts == 0);

	const size_t rowsPerNode = rows / numNodes;

	for (size_t i = 0; i < rows; i += rowsPerNode) { // loop nodes

		const int nodeid = i / rowsPerNode;

		#pragma oss task weakout(ret[i * cols; rowsPerNode * cols]) \
			node(nodeid) label("initalize_weak")
		{
			for (size_t j = i; j < i + rowsPerNode; j += ts) { // loop tasks

				#pragma oss task out(ret[j * cols; ts * cols]) \
					node(nanos6_cluster_no_offload) label("initalize_slice")
				{
					struct drand48_data drand_buf;
					srand48_r(j, &drand_buf);
					double x;

					const size_t elems = ts * cols;

					for (size_t k = 0; k < elems; ++k) {
						drand48_r(&drand_buf, &x);
						ret[j * cols + k] = x;
					}
				}
			}
		}
	}
}


static inline void matmul_base(const double *A, const double *B, double * const C,
                               size_t lrowsA, size_t dim, size_t colsBC)
{
	for (size_t i = 0; i < lrowsA; ++i) {
		for (size_t k = 0; k < colsBC; ++k)
			C[i * colsBC + k] = 0.0;

		for (size_t j = 0; j < dim; ++j) {
			const double temp = A[i * dim + j];

			for (size_t k = 0; k < colsBC; ++k) {
				C[i * colsBC + k] += (temp * B[j * colsBC + k]);
			}
		}
	}
}

static inline void __print(const double * const mat,
                           const size_t rows, const size_t cols,
                           const char prefix[64], const char name[64]
) {
	#pragma oss task in(mat[0; rows * cols]) label("matrix_print")
	{
		char filename[256];
		sprintf(filename,"%s_%s.mat", prefix, name);
		FILE *fp = fopen(filename, "w+");
		assert(fp);

		fprintf(fp, "# name: %s\n", name);
		fprintf(fp, "# type: matrix\n");
		fprintf(fp, "# rows: %lu\n", rows);
		fprintf(fp, "# columns: %lu\n", cols);

		for (size_t i = 0; i < rows; ++i) {
			for(size_t j = 0; j < cols; ++j) {
				fprintf(fp, "%3.8lf ", mat[i * cols + j]);
			}
			fprintf(fp,"\n");
		}
		fclose(fp);
	}
}

#define printmatrix_task(mat, rows, cols, prefix)	\
	__print(mat, rows, cols, prefix, #mat)

#endif // MATVEC_H
