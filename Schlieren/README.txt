For copyright info, see LICENSE.txt


Schlieren README:

the program was set up with a series of #ifdefs to handle a large number of options very fast without impeding performance with if statements (and because I suck at C programming)

pretty much all of the options you will want to tweak are listed in simpleGL.cu in the top of the file.

Note that memory is a huge problem:  each option may use up more memory. 

#define PRECOMPUTE_GRADIENT 0
#define INTERPOLATE_GRADIENT 1  // must also precompute for this
#define TRILINEAR_INTERPOLATION 1 // for data
#define GUASSIAN_FILTER 0  //note guassianfilter isn't thread safe
#define PROJECT_TO_FILM 1 // project rays to the film plane, otherwise they are taken as soon as they hit the end 
#define PROJECTION_DISTANCE 0.2f

these all affect how rays are traced through the volume.  Precomputing gradient(normals) will compute the gradient information before rendering.  This uses a lot more memory so it really only makes sense if you are also 
INTERPOLATE_GRADIENT for smoothing gradient values.
TRILINEAR_INTERPOLATION will interpolate data values of the data volume at runtime.  This doesn't include gradient(normal) values.  

GUASSIAN_FILTER is a filtering technique which weights rays stronger that are closer to the middle of the pixel than the edge.  Useful if you are seeing a lot of fine grained noise.  
PROJECT_TO_FILM will project rays to the film plane, instead of just stopping the at the end of the volume.  PROJECTION_DISTANCE is the distance from the center of the volume.  note that if this is smaller than the volume itself some rays may not be moved.  



#define SIN_DATASET 0
#define VASE_DATASET 0
#define TSI_DATASET 0 
#define COMBUSTION_DATASET 1
#define PV_DATASET 0
  #define PV_DATASET_MAG 0
  #define PV_DATASET_DENSITY 0
  #define PV_DATASET_PRESSURE 0
#define OCEAN_DATASET 0
#define DATA_SIZE 256.0f
#define STEP_SIZE 0.001f
#define NUM_RAYS 200000.0f  // number of rays to trace each frame
#define NUM_PER_THREAD 1 //trace multiple rays in each kernel.  only do if you are tracing more rays
#define START_TIMESTEP 42
#define END_TIMESTEP 45 // end timestep will be 1 past, ie timestep of 70 will load 0-69
#define NUM_TIMESTEPS (END_TIMESTEP-START_TIMESTEP)
#define SAMPLES_PER_PIXEL 10000  //number of rays to trace through before going to next timestep
#define LOOP 0

thse above defines termine datasets behavior.  make sure only on dataset is active at once.  sine wave and vase are computed on the fly datasets, determined by DATA_SIZE.  the other datasets are loaded from files.  If you want to modify what's loaded do a search for COMBUSTION_DATASET for example and it will set the data sizes.  They each are different.  If you are having memory problems look up where those datasets set their datasizes and lower data_depth.

Animation:  for animation you have start_timestep and end_timestep.  note that for an animation you will also want to set CLAMP, and play with NUM_RUNS so it's smooth enough and NORMALIZATION_CONSTANT so that it is bright/dark enough (this is like adjusting aperture).#define GLOBAL_MINMAX 0 will set the color information for the entire timesteps instead of coloring based on the min/max values of each individual timesteps.  Each timestep is saved in images/render%d.png .  I used ffmpeg to make movies.  

#define COLOR 1
#define COLOR_DIRECTION 0
#define COLOR_ACCUMULATE 1
#define COLOR_ON_OTHER_DATA 0

these set coloring instead of grayscale.  you may also have to look in simpleGL_kernel.cu to adjust the "transfer function" for the coloring.  If you don't see anything on the screen one thing to try is turning off coloring to make sure that it isn't being colored black.  COLOR_DIRECTION colors rays based on the direction they were refracted.  COLOR_ACCUMULATE colors based on data index and COLOR_ON_OTHER_DATA will color the data based on a second data index (huge memory consumed!).  


