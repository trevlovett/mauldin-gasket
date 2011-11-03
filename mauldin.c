#include <assert.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// app constants
#define NUM_THREADS 4
#define N 200000000LL / NUM_THREADS // fast
//#define N 10000000000LL / NUM_THREADS // produces nicely detailed image
#define NX 4000 // image width (pixels)
#define NY 4000 // image height (pixels)
#define SCALE (NX / 2)

// math constants
#define SQRT3 1.73205081f
#define SQRT1p5 1.22474487f

// each thread receives one of these
typedef struct thread_context {
    unsigned int id;
    float *image;
    // used for locking image table when updating (default: not used)
    pthread_mutex_t *image_mutex;
} thread_context;

inline void plot(float *image, float x, float y, float luma) {
    int ix, iy, offset;
    ix = (int)(x*SCALE + NX*0.5f);
    iy = (int)(y*SCALE + NY*0.5f);

    if (luma < 0)
        luma = 0.0f;

    if (ix >=0 && iy >= 0 && ix < NX && iy < NY) {
       offset = NX*iy + ix;
       // accumulating lumas produces nicely shaded image
       image[offset] += luma;
    }
}

void *mauldin(void *context) {
    float x[3], y[3], xa, ya, xb, yb, xc, yc, fact, x_exact, y_exact;
    float btl, btr, bbl, bbr, fx, fy;
    int choice;
    thread_context *t_context = (thread_context *)context;
    float *image = t_context->image;
    unsigned long long i;

    // use unique thread id as srand seed
    srand(t_context->id);

    // random initial conditions
    x[0] = (float)rand()/(float)RAND_MAX;
    y[0] = (float)rand()/(float)RAND_MAX;

    // don't render first 100 iterations
    for (i = 0; i < 100; i++) {
        choice = rand_r(&(t_context->id))%3;

        x[1] = (-0.5f)*x[0] - 0.866025404f*y[0]; // xr == 1
        y[1] = (-0.5f)*y[0] + 0.866025404f*x[0];

        x[2] = x[1]*x[1] - y[1]*y[1];          // x2 == 2
        y[2] = 2.0f * x[1] * y[1];

        xa = (SQRT3-1.0f)*x[choice]+1.0f;
        ya = (SQRT3-1.0f)*y[choice];

        xb = -x[choice]+(SQRT3+1.0f);
        yb = -y[choice];

        fact = 1.0f / (xb*xb+yb*yb);
        xc = xb * fact;
        yc = -yb * fact;

        x[0] = (xa*xc - ya*yc) * SQRT1p5;
        y[0] = (xa*yc + xc*ya) * SQRT1p5;
    }

    for (i = 100; i < N; i++) {
        choice = rand_r(&(t_context->id))%3;

        x[1] = (-0.5f)*x[0] - 0.866025404f*y[0]; // xr == 1
        y[1] = (-0.5f)*y[0] + 0.866025404f*x[0];

        x[2] = x[1]*x[1] - y[1]*y[1];          // x2 == 2
        y[2] = 2.0f * x[1] * y[1];

        xa = (SQRT3-1.0f)*x[choice]+1; 
        ya = (SQRT3-1.0f)*y[choice];

        xb = -x[choice]+(SQRT3+1.0f); 
        yb = -y[choice];

        fact = 1.0f / (xb*xb+yb*yb);
        xc = xb * fact;             
        yc = -yb * fact;

        x[0] = (xa*xc - ya*yc) * SQRT1p5;
        y[0] = (xa*yc + xc*ya) * SQRT1p5;

        // bilinear interp
        fx = x[0] - (int)x[0];
        fy = y[0] - (int)y[0];
        btl = (1.0f - fx) * (1.0f - fy);
        btr = fx * (1.0f - fy);
        bbl = (1.0f - fx) * fy;
        bbr = fx * fy;

        // Uncomment lock/unlock if you would like to be "correct" and avoid
        // interference.  Probability of interference is quite low, however,
        // and using mutexes slows things down considerably.
        //pthread_mutex_lock(t_context->image_mutex);
        plot(image, x[0],   y[0],   btl);
        plot(image, x[0]+1, y[0],   btr);
        plot(image, x[0],   y[0]+1, bbl);
        plot(image, x[0]+1, y[0]+1, bbr);
        //pthread_mutex_unlock(t_context->image_mutex);
    }

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    pthread_t threads[NUM_THREADS];
    pthread_mutex_t image_mutex;
    thread_context thread_contexts[NUM_THREADS];
    FILE *outfile=NULL;
    int rc, x, y, offset, k;
    long t;
    float max = 0, logmax = 0;
    unsigned int luma, num_image_elems = NX*NY;
    float *image = (float *)malloc(num_image_elems*sizeof(float));

    // initialize mutex
    pthread_mutex_init(&image_mutex, NULL);

    clock_t start = clock();

    for (t=0; t < NUM_THREADS; t++) {
        thread_contexts[t].id = t;
        thread_contexts[t].image = image;
        thread_contexts[t].image_mutex = &image_mutex;

        fprintf(stderr, "\nSpawning thread %ld", t);
        rc = pthread_create(&threads[t], NULL, mauldin, (void *)&thread_contexts[t]);

        if (rc) {
            fprintf(stderr, "\nERROR; return code from pthread_create() is %d", rc);
            exit(-1);
        }
    }

    fprintf(stderr, "\nProcessing...");

    for (t=0; t < NUM_THREADS; t++) {
        rc = pthread_join(threads[t], NULL);
        assert(rc == 0);
        fprintf(stderr, "\nThread %ld finished.", t);
    }
    clock_t end = clock();
    float cpu_time = ((float)(end - start))/((float)(CLOCKS_PER_SEC));

    fprintf(stderr, "\nCLOCKS_PER_SEC: %ld", CLOCKS_PER_SEC);
    fprintf(stderr, "\nclocks elapsed: %ld", (end - start));
    fprintf(stderr, "\ncpu_time: %lf", cpu_time);
    fprintf(stderr, "\niterations_per_sec: %e",(((float)N*NUM_THREADS)/cpu_time));

    fprintf(stderr, "\n\nFinding maximum luminosity..");
    for (k = 0; k < num_image_elems; k ++) {
        if (image[k] > max)
            max = image[k];
    }

    logmax = log(max + 1);

    fprintf(stderr, "\nmax = %f, log(max + 1) = %f", max, logmax);
    fprintf(stderr, "\nWriting image to file...");
    // write image to PPM file
    outfile = fopen("output.ppm", "w");
    fprintf(outfile, "P2\n");
    fprintf(outfile, "%d %d\n", NX, NY);
    fprintf(outfile, "65535");
    for (y = 0; y < NY; y++) {
        fprintf(outfile, "\n");
        for (x = 0; x < NX; x++) {
            offset = NX*y+x;
            luma = (unsigned int)(65535.0 * log(image[offset] + 1.0) / logmax);
            assert(luma >= 0 && luma <= 65535);
            fprintf(outfile, "%d ", luma);
        }
    }

    fprintf(stderr, "Done.\n");

    free(image);
    close(outfile);
    pthread_mutex_destroy(&image_mutex);
    pthread_exit(NULL);
}
