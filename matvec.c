/*
 * Copyright (C) 2021  Jimmy Aguilar Mena
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#include "tutorial.h"

void matvec_tasks_strong(const double *A, const double *B, double *C,
                         size_t ts, size_t dim
) {
	printf("# matvec_strong_flat\n");
	assert(ts <= dim);
	assert(dim % ts == 0);

	const size_t numNodes = nanos6_get_num_cluster_nodes();
	const size_t rowsPerNode = dim / numNodes;

	for (size_t i = 0; i < dim; i += ts) {
		int nodeid = i / rowsPerNode;

		#pragma oss task in(A[i * dim; ts * dim])	\
			in(B[0; dim])							\
			out(C[i; ts])							\
			node(nodeid) label("strongmatvec")
		matmul_base(&A[i * dim], B, &C[i], ts, dim, 1);
	}
}


void matvec_tasks_weak(const double *A, const double *B, double *C,
                       size_t ts, size_t dim
) {
	printf("# matvec_weak_node\n");
	assert(ts <= dim);
	assert(dim % ts == 0);

	const size_t numNodes = nanos6_get_num_cluster_nodes();

	const size_t rowsPerNode = dim / numNodes;
	assert(ts <= rowsPerNode);
	assert(rowsPerNode % ts == 0);

	for (size_t i = 0; i < dim; i += rowsPerNode) {

		const int nodeid = i / rowsPerNode;

		#pragma oss task weakin(A[i * dim; rowsPerNode * dim])			\
			weakin(B[0; dim])											\
			weakout(C[i; rowsPerNode])									\
			node(nodeid) label("weakmatvec")
		{
			#pragma oss task in(A[i * dim; rowsPerNode * dim])			\
				in(B[0; dim])											\
				out(C[i; rowsPerNode])									\
				node(nanos6_cluster_no_offload) label("fetchtask")
			{
				// This is a fetch task.
			}

			for (size_t j = i; j < i + rowsPerNode; j += ts) {
				#pragma oss task in(A[j * dim; ts * dim])				\
					in(B[0; dim])										\
					out(C[j; ts])										\
					node(nanos6_cluster_no_offload) label("strongmatvec")
				matmul_base(&A[j * dim], B, &C[j], ts, dim, 1);
			}
		}
	}
}


int main(int argc, char* argv[])
{

	const char *PREFIX = basename(argv[0]);
	const int dim = atoi(argv[1]);
	const int TS = atoi(argv[2]);

	printf("# Initializing data\n");

	const size_t dim2 = dim * dim;

	double *A = (double *) nanos6_dmalloc(dim * dim * sizeof(double),
	                                      nanos6_equpart_distribution, 0, NULL);
	double *B = (double *) nanos6_dmalloc(dim * sizeof(double),
	                                      nanos6_equpart_distribution, 0, NULL);
	double *C = (double *) nanos6_dmalloc(dim * sizeof(double),
	                                      nanos6_equpart_distribution, 0, NULL);

	init(A, dim, dim, TS);    // this initialized by blocks ts x rows
	init(B, dim, 1, TS);    // this splits the array in ts
	#pragma oss taskwait

	printf("# Starting algorithm\n");

	matvec_tasks_weak(A, B, C, TS, dim);
	#pragma oss taskwait

	printf("# Finished algorithm...\n");

	printmatrix_task(A, dim, dim, "matvec");
	printmatrix_task(B, dim, 1, "matvec");
	printmatrix_task(C, 1, dim, "matvec");

	#pragma oss taskwait

	nanos6_dfree(A, dim2 * sizeof(double));
	nanos6_dfree(B, dim * sizeof(double));
	nanos6_dfree(C, dim * sizeof(double));

	return 0;
}
