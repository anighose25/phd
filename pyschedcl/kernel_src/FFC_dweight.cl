// A X B = (M * K , K * N)

__kernel void myGEMM3(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C,
                      __global float* UpdateWeight,
                      __global float* UpdateBiased) {

    int RTS=TS/WPT;
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS/WPT == RTS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
    // Initialise the accumulation registers
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }

    // Loop over all tiles
    const int numTiles = K/TS;
    //printf("new %d %d %d %d %d\n" , numTiles , N , M , K , TS);
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int w=0; w<WPT; w++) {
            const int tiledRow = TS*t + row;
            const int tiledCol = TS*t + col;
            Asub[row + w*RTS][col] = A[(tiledCol ) + (globalRow+ w*RTS)*K];
            Bsub[row + w*RTS][col] = B[(globalCol ) + (tiledRow+ w*RTS)*N];
            //printf("%f %f %d\n", A[(tiledCol + w*RTS) + globalRow*K] , B[(globalCol + w*RTS) + tiledRow*N] , (globalCol + w*RTS) + tiledRow*N);
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            for (int w=0; w<WPT; w++) {
                acc[w] += Asub[row+w*RTS][k] * Bsub[k][col];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    //printf("%d %d %d %d \n" , (globalCol )*M + globalRow , globalRow , globalCol  , M);
    for (int w=0; w<WPT; w++) {
        C[(globalCol) + (globalRow + w*RTS )*N] = acc[w];
        UpdateWeight[(globalCol) + (globalRow + w*RTS )*N] -= LR*C[(globalCol) + (globalRow + w*RTS )*N];
    }
    
    #pragma omp atomic
        UpdateBiased[(globalCol)] -= LR*B[(globalCol) + (globalRow)*N]/N;
    
}
