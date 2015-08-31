/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2012-2013
   Scientific Computing and Imaging Institute, University of Utah

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
   */





#define BLUR_DATA 0
#define GAUSSIAN 1  //guassian or mean


#include "opengl_include.h"
//#include "cutil.h"
//#include "cutil_math.h"
//#include <helper_cuda.h>
//#include <helper_math.h>
#include "cudaIncludes.h"
#include <teem/nrrd.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "main.h"
#include "schlierenrenderer.h"

#include <float.h>
#include <assert.h>
//#include <mpi.h>
using namespace std;


float data_fudge = 1.0f;

SchlierenRenderer renderer;

void display()
{
  cout << "glut display\n";
  static int count = 0;
  if (count++ >= 1000)
  {
    cout << "finished rendering\n";
    return;

  }
  renderer.render();
  glutPostRedisplay();
  glutSwapBuffers();
}

void keyboard( unsigned char key, int x, int y) {}
double last_x = 0.0;
double last_y = 0.0;
bool dirty_gradient = true;
bool alteringTF = false;
bool rotating = false;
void mouse(int button, int state, int x, int y) {
  if (button == GLUT_RIGHT_BUTTON) {
#if USE_TF
    if (state == GLUT_DOWN)
    {
      alteringTF = true;
      dirty_gradient = true;
      last_x = x;
      last_y = y;
    }
    else {
      //cudaMemcpy(d_TFBuckets, transferFunction->buckets, sizeof(float4)*transferFunction->numBuckets, cudaMemcpyHostToDevice);
      alteringTF = false;
    }
#endif
  }
  else if (button == GLUT_LEFT_BUTTON) {
    last_x = x;
    last_y = y;
    if (state == GLUT_DOWN) {
      rotating = true;
    }
    else {
      rotating = false;
      renderer.clear();
    }
  }
}
double rot_x = 0;
double rot_y = 0;
void motion(int x, int y)
{
#if USE_TF
  if (alteringTF) {
    for(int i =0 ; i < 100; i++) {
      int tx = last_x + (x - last_x)*(i/100.0f);
      int ty = last_y + (y - last_y)*(i/100.0f);
      if (tfRenderable->Contains(tx,window_height - ty))
        tfRenderable->ProcessMouse(tx,window_height -ty);
    }
    clear();
  }
#endif
  float scale = 0.00001;
  if (rotating) {
    rot_x += (x-last_x)*scale;
    rot_y += (y-last_y)*scale;

    //setCamera();
    renderer.rotate(rot_x, rot_y);
    renderer.updateCamera();

    //    printf("setting camera: %f %f %f\n", camera_pos.x, camera_pos.y, camera_pos.z);
    //    printf("magnitude of camera from center: %f\n", float(magnitude(minus(camera_pos, center))));
    //    printf("camera rotx roty: %f %f\n", rot_x, rot_y);
    renderer.clear();
  }
  last_x = x;
  last_y = y;
}
bool initGL()
{
  return true;
}

void reshape(int w, int h)
{
  renderer.setRenderSize(w,h);
}


/*template<class T, int N> convolve(T* data_in, T* data_out, float* kernel, float* kernel_width, size_t stride, size_t steps)
  {
  T* din = data_in;
  T* dout = data_out;
  for(size_t i = 0; i < steps; i++) {

  }
  }*/

#define UNIVERSALGASCONSTANT    8.314472                // J/(mol.K) = BOLTZMANNSCONSTANT * AVOGADROSNUMBER

// Ciddor's formulation
extern "C" double   AirRefractiveIndex(double temperature = 293.15, double pressure = 101325.0, double wavelength = 580.0, double relativeHumidity = 0.0, double co2ppm = 450.0);

// Simpler formulation
extern "C" double   AirRefractionIndex(double temperature = 293.15, double pressure = 101325.0, double wavelength = 580.0);


extern "C" void loadNRRD(DataFile* datafile, int data_min, int data_max);


int main(int argc, char** argv)
{
  bool record = false;
  float step_size = 0.1;
  int data_min = 0;
  int data_max = 1024;
  bool temp = false;
  bool press = false;
  bool convert = false;
  string filename = "/home/sci/brownlee/data/ExxonMobil/faults.nrrd";
  vector<string> files, tempFiles, pressFiles;
  DataFile* dataFiles[200];
  /*if (argc > 1) {
    filename = string(argv[1]);
    }
    if (argc > 2) {
    stringstream s(argv[2]);
    s >> data_min;
    }
    if (argc > 3) {
    stringstream s(argv[3]);
    s >> data_max;
    }*/
  for(int i = 1; i < argc; i++) {
    string arg(argv[i]);
    if (arg == "-minz") {
      stringstream s(argv[++i]);
      s >> data_min;
    } else if (arg == "-record") {
      record = true;
    } else if (arg == "-maxz") {
      stringstream s(argv[++i]);
      s >> data_max;
    } else if (arg == "-fudge") {
      stringstream s(argv[++i]);
      s >> data_fudge;
    } else if (arg == "-convert") {
      convert = true;
    } else if (arg == "-pressure") {
      press = true;
      temp = false;
    } else if (arg == "-temperature") {
      temp = true;
      press = false;
    } else if (arg == "-step_size") {
      stringstream s(argv[++i]);
      s >> step_size;
    } else {  //assume that it is a nrrd file...
      if (temp)
        tempFiles.push_back(string(argv[i]));
      else if (press)
        pressFiles.push_back(string(argv[i]));
      else
        files.push_back(string(argv[i]));
    }
  }
  if (files.empty())
  {
    cerr << "no input files specified\n";
    //   exit(1);
  }
  if (convert) {
    for(int filenum = 0; filenum < tempFiles.size(); filenum++) {
      printf("loading file %s : ", tempFiles[filenum].c_str());
      Nrrd* tempNrrd = nrrdNew();
      Nrrd* pressNrrd = nrrdNew();
      bool usePress = false;
      if(nrrdLoad(tempNrrd, tempFiles[filenum].c_str(), 0)) {
        char* err=biffGetDone(NRRD);
        cerr << "Failed to open \"" + string(tempFiles[filenum].c_str()) + "\":  " + string(err) << endl;
        exit(__LINE__);
      }
      if (filenum < pressFiles.size())
        usePress = true;
      if (usePress) {
        if(nrrdLoad(pressNrrd, pressFiles[filenum].c_str(), 0)) {
          char* err=biffGetDone(NRRD);
          cerr << "Failed to open \"" + string(pressFiles[filenum].c_str()) + "\":  " + string(err) << endl;
          exit(__LINE__);

        }
      }
      int sizex, sizey, sizez;
      sizex = tempNrrd->axis[0].size;
      sizey = tempNrrd->axis[1].size;
      sizez = tempNrrd->axis[2].size;
      printf(" size: %f %f %f ", float(sizex), float(sizey), float(sizez));
      float* data = new float[sizex*sizey*sizez];
      float min = FLT_MAX;
      float max = -FLT_MAX;
      for(int i = 0; i < sizex; i++) {
        for(int j = 0; j < sizey; j++) {
          for( int k = 0; k < sizez; k++) {
            //assume it's unsigned int... but needs to change
            size_t index = i + j*sizex + k*sizex*sizey;
            double temp = (((float*)tempNrrd->data)[index]);
            if (usePress) {
              double press = (((float*)pressNrrd->data)[index]) + 101325.0;

              (((float*)tempNrrd->data)[index]) = AirRefractionIndex(temp, press);
            } else {
              (((float*)tempNrrd->data)[index]) = AirRefractionIndex(temp);
            }
          }
        }
      }
      char outFilename[512];
      sprintf(outFilename, "airIR_%05d.nrrd", filenum);
      nrrdSave(outFilename, tempNrrd, NULL);
      nrrdNuke(tempNrrd);
      nrrdNuke(pressNrrd);
      delete[] data;
    }
    return 0;
  }

  //TODO:remove hardcoded combstuion stuff
  //  files.push_back("/home/falcor/data/combustion/jet_chi_0060.nhdr");
  //  files.push_back("/home/falcor/data/combustion/jet_hr_0060.nhdr");
  //  files.push_back("/home/falcor/data/combustion/jet_Y_OH_0060.nhdr");
  //  files.push_back("/home/falcor/data/combustion/jet_mixfrac_0060.nhdr");
  //  files.push_back("/home/falcor/data/combustion/jet_vort_0060.nhdr");
  //  files.push_back("/home/falcor/data/combustion/jet_mixfrac_0060_resampled.nrrd");
  //
  for(int file = 0; file < files.size(); file++) {
    dataFiles[file] = new DataFile();
    dataFiles[file]->filename = new char[256];
  }
  char** input_files;
  input_files = new char*[files.size()];
  for( int i = 0; i < files.size(); i++) {
    cout << "file: " << files[i] << endl;
    input_files[i] = new char[files[i].length()];
    strcpy(input_files[i], files[i].c_str());
    strcpy(dataFiles[i]->filename, input_files[i]);
    loadNRRD(dataFiles[i],data_min, data_max);
  }

  float* data = dataFiles[0]->data;
  int zrange = dataFiles[0]->sizez;
  //
  // multivariate data
  //
  //    int zrange = 120;
  //if (zrange > data_max - data_min)
  //    zrange = data_max - data_min;
  //  data = new float[480*720*zrange];
  //
  //  float4* color = new float4[480*720*zrange];
  //
  //  for(int i = 0 ; i < 480; i++){
  //      for(int j =0;j<720; j++) {
  //          for(int k =0; k < zrange; k++) {
  //              size_t index = k*720*480 + j*480 + i ;
  //              if (index >= 480*720*zrange){
  //                  cerr << "index out of bounds\n";
  //                  continue;
  //              }
  //
  //            data[index] = (0
  //                           + dataFiles[0]->data[index] * 1.0/31538.289062
  //                            + (dataFiles[1]->data[index] + 19524868096.0) * 1.0/19524868096.0
  //                            + dataFiles[2]->data[index]*1.0/0.001714
  //                            + dataFiles[3]->data[index]
  //                            + dataFiles[4]->data[index]*1.0/876405.500000
  //                            )/5.0;
  //
  //	     float4 c1,c2,c3,c4,c5;
  //	     float scale = 10.0;
  //	    c1 = make_float4(1,0,0,1)*scale;
  //	    c2 = make_float4(0,1,0,1)*scale;
  //	    c3 = make_float4(0,0,1,1)*scale;
  //	    c4 = make_float4(.5,.5,0,1)*scale;
  //	    c5 = make_float4(0,.5,.5,1)*scale;
  //	    color[index] = (make_float4(0,0,0,1)
  //	                    + c1 * dataFiles[0]->data[index] * 1.0/31538.289062
  //                            + c2 * (dataFiles[1]->data[index] + 19524868096.0) * 1.0/19524868096.0
  //                            + c3 * dataFiles[2]->data[index]*1.0/0.001714
  //                            + c4 * dataFiles[3]->data[index]
  //                            + c5 * dataFiles[4]->data[index]*1.0/876405.500000
  //			    );
  //          }
  //      }
  //   }

  //
  //  end multivariate data
  //

  cout << "setting up renderer\n";
  renderer.setData(data, dataFiles[0]->sizex,dataFiles[0]->sizey,zrange);
  renderer.setStepSize(step_size);
  //  renderer._params.color_data = color;
  //  renderer.setFilter(new SchlierenShadowgraphCutoff());
  //  float* cutoffData = new float[256*256*4];
  //  float* datap = cutoffData;
  //  for(int i = 0; i < 256; i++)
  //  {
  //      for(int j = 0; j < 256; j++)
  //      {
  //            datap[0] = float(i)/256.0;
  //            datap[1] = float(i)/256.0;
  //            datap[2] = float(i)/256.0;
  //            datap[3] = 1.0f;
  //          datap+=4;
  //      }
  //  }
  //renderer.setFilter(new SchlierenImageCutoff(cutoffData));
  //renderer.setFilter(new SchlierenInterforemetryCutoff());
  cout << "setting filter\n";
  renderer.setFilter(new SchlierenPositiveHorizontalKnifeEdgeCutoff());
  // renderer.setFilter(new SchlierenShadowgraphCutoff());
  cout << "setting image filter\n";
  renderer.setImageFilter(new ImageFilter());
  cout << "setting up glut\n";

  glutInit( &argc, argv);
  glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize( renderer.getWidth(), renderer.getHeight());
  glutCreateWindow( "CD CUDA Schlieren");

  // initialize GL
  if( false == initGL()) {
    exit(1);
  }

  // register callbacks
  glutDisplayFunc( display);
  glutKeyboardFunc( keyboard);
  glutMouseFunc( mouse);
  glutMotionFunc( motion);
  glutReshapeFunc(reshape);

  cout << "mainloop\n";
  // start rendering mainloop
  glutMainLoop();
  cout << "mainloop done\n";

  // MPI_Init(&argc, &argv);
  int rank, size;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //   MPI_Comm_size(MPI_COMM_WORLD, &size);
  rank = 0;
  size = 1;
  // run(argc, argv, rank, size, 0, 0, 0, input_files, files.size(), data_min, data_max, dataFiles, step_size, record);
  //    MPI_Finalize();
  return 0;
}





extern "C"
{
  void loadNRRD(DataFile* datafile, int data_min, int data_max)
  {

    printf("loading file %s : ", datafile->filename);
    Nrrd* nrrd = nrrdNew();
    if(nrrdLoad(nrrd, datafile->filename, 0)) {
      char* err=biffGetDone(NRRD);
      cerr << "Failed to open \"" + string(datafile->filename) + "\":  " + string(err) << endl;
      exit(__LINE__);
    }
    int sizex, sizey, sizez;
    sizex = nrrd->axis[0].size;
    sizey = nrrd->axis[1].size;
    sizez = nrrd->axis[2].size;
    if (data_max > sizez)
      data_max = sizez;
    if (sizez > (data_max-data_min))
      sizez = data_max-data_min;
    printf(" size: %f %f %f ", float(sizex), float(sizey), float(sizez));
    float* data = new float[sizex*sizey*sizez];
    float min = FLT_MAX;
    float max = -FLT_MAX;
    float* dataNrrd = (float*)nrrd->data;
    float* datai = data;
    for(int i = 0; i < sizex; i++) {
      for(int j = 0; j < sizey; j++) {
        for( int k = 0; k < sizez; k++) {
          *datai = (*dataNrrd)*data_fudge;

          if (*datai > max)
            max = *datai;
          if (*datai < min)
            min = *datai;
          datai++;
          dataNrrd++;

        }
      }
    }


    datafile->data = data;
    datafile->sizex = sizex;
    datafile->sizey = sizey;
    datafile->sizez = sizez;
    nrrdNuke(nrrd);
    printf("  ...done\n");
  }


  // --------------------------------------------------------------------------------------
  // -- Return the compressibility
  // -- temperature                               : Kelvin
  // -- pressure                                  : Pascal
  // -- waterVaporMolarFraction   : [0-1]
  // --------------------------------------------------------------------------------------
  double Compressibility(double temperature, double pressure, double waterVaporMolarFraction)
  {
    double a0, a1, a2, b0, b1, c0, c1, d, e, z, pt, tC;

    a0 = 1.58123e-6;
    a1 = -2.9331e-8;
    a2 = 1.1043e-10;
    b0 = 5.707e-6;
    b1 = -2.051e-8;
    c0 = 1.9898e-4;
    c1 = -2.376e-6;
    d  = 1.83e-11;
    e  = -0.765e-8;

    pt = pressure / temperature;
    tC = temperature - 273.15;      // Temperature in Celcius

    z = 1.0 + pt * (pt * (d + e*waterVaporMolarFraction*waterVaporMolarFraction)
        - (a0 + (a1 + a2*tC)*tC + ((b0 + b1*tC) + (c0 + c1*tC)*waterVaporMolarFraction) * waterVaporMolarFraction));

    return z;
  }


  // --------------------------------------------------------------------------------------
  // -- Compute the dryAirComponent and waterVaporComponent of the density
  // -- temperature                               : Kelvin
  // -- pressure                                  : Pascal
  // -- waterVaporMolarFraction   : [0-1]
  // -- co2ppm                                    : parts per million
  // --------------------------------------------------------------------------------------
  void Density(double temperature, double pressure, double waterVaporMolarFraction, double co2ppm, double * dryAirComponent, double * waterVaporComponent)
  {
    double pzrt, Ma, Mw, z;

    Mw = 0.018015;                                                                  // Molar mass of water vapor
    Ma = 0.0289635 + 12.011e-9 * (co2ppm - 400.0);  // Molar mass of dry air containing co2 ppm

    z = Compressibility(temperature, pressure, waterVaporMolarFraction);

    pzrt = pressure / (z * UNIVERSALGASCONSTANT * temperature);

    if (dryAirComponent)
      *dryAirComponent                = pzrt * Ma * (1.0 - waterVaporMolarFraction);

    if (waterVaporComponent)
      *waterVaporComponent    = pzrt * Mw * (      waterVaporMolarFraction);
  }


  // --------------------------------------------------------------------------------------
  // -- Return the (refractive index of air - 1) for the given parameters
  // -- temperature               : Kelvin
  // -- pressure                  : Pascal
  // -- wavelength                : nanometer
  // -- relativeHumidity  : [0-1]
  // -- co2ppm                    : parts per million
  // --------------------------------------------------------------------------------------
  double AirRefractiveIndex(double temperature, double pressure, double wavelength, double relativeHumidity, double co2ppm)
  {
    // Saturation vapor pressure of water vapor in air
    double svp = exp((1.2378847e-5*temperature - 1.9121316e-2)*temperature + 33.93711047 - 6.3431645e3/temperature);

    // Enhancement factor of water vapor in air
    double f, tC = temperature - 273.15;
    f = 1.00062 + 3.14e-8*pressure + 5.6e-7*tC*tC;

    // Molar fraction of water vapor
    double xw = relativeHumidity * f * svp / pressure;

    double paxs, pws, pa, pw;
    Density(     288.15, 101325.0, 0.0, co2ppm, &paxs, NULL);       // Density of standard dry air
    Density(     293.15,   1333.0, 1.0, co2ppm,  NULL, &pws);       // Density of standard water vapor
    Density(temperature, pressure,  xw, co2ppm,   &pa,  &pw);       // Density of moist air

    double waveNumber, waveNumber2;
    waveNumber = 1000.0 / wavelength;       // nanometer to micrometer
    waveNumber2 = waveNumber * waveNumber;

    // Refractivity of standard air (15 C, 101325 Pascal, 0% humidity, 450 ppm of CO2)
    double nas1 = (5792105.0 / (238.0185  - waveNumber2) + 167917.0 / (57.362  - waveNumber2)) * 1.0e-8;

    // Refractivity of standard air with co2 ppm
    double naxs1 = nas1 * (1.0 + 0.534e-6 * (co2ppm - 450.0));

    // Refractivity of standard water vapor (20 C, 1333 Pascal, 100% humidity)
    double nws1 = 1.022e-8 * (295.235 + (2.6422 - (0.03238 + 0.004028*waveNumber2) * waveNumber2) * waveNumber2);

    return naxs1 * pa / paxs + nws1 * pw / pws;
  }


  // --------------------------------------------------------------------------------------
  // -- Return (refractiveIndex - 1)
  // -- temperature               : Kelvin
  // -- pressure                  : Pascal
  // -- wavelength                : meter
  // --------------------------------------------------------------------------------------
  double AirRefractionIndex(double temperature, double pressure, double wavelength)
  {
    double tempC, sigma, index;

    tempC = temperature - 273.15;
    sigma = 1.0e-6 / wavelength;
    index = 0.0472326 / (173.3 - sigma * sigma);
    index = index * pressure * (1.0 + pressure * (60.1 - 0.972 * tempC) * 1e-10) / (96095.43 * (1.0 + 0.003661 * tempC));

    return index;
  }

}
