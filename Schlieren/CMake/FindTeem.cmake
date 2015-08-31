
##################################################################
# Look for Teem http://teem.sourceforge.net/

# 
# The script should work with TEEM-1.9 which includes a CMake build
# along with scripts to locate teem libraries. The scripts appear to
# included only in the source build. 
#
# TEEM 1.9 also uses separate libraries for all of the different teem
# modules; there is no monolithic libteem.{so,a}
# 
# The following relevant variables are introduced by this script:
#
# TEEM_LIBRARY_DIRS -- Teem library directory. (from TEEMConfig)
# TEEM_LIBRARY      -- Path to Teem library    (local)
# TEEM_INCLUDE_DIRS -- Include directory.      (from TEEMConfig)
#
# Note that this script does not include the TEEMUse.cmake script which 
# appears to modify the global compiler and linker command line options.
# These options are specifically tuned for many compilers in Manta.
# 

# First locate the unu command in the path.
FIND_PATH(FOUND_TEEM_BIN unu
  DOC "Location of teem binaries (like unu)"
  )

# Check to see if unu/teem is in the path.
IF (FOUND_TEEM_BIN)

  # Search for TEEMConfig using a relative path
  FIND_FILE(FOUND_TEEMCONFIG_CMAKE
    TEEMConfig.cmake
    ${FOUND_TEEM_BIN}/../lib/TEEM-1.10
    )
  
  # Include the teem configuration.
  IF (FOUND_TEEMCONFIG_CMAKE)

    # Include a generated configure script
    INCLUDE(${FOUND_TEEMCONFIG_CMAKE})

    # Enable found flag.
    SET(FOUND_TEEM  TRUE)

    # Add the include directory to the build
    SET(TEEM_INCLUDE_DIRS "" CACHE PATH "Path for teem include, if required.")
    INCLUDE_DIRECTORIES(${TEEM_INCLUDE_DIRS})

    # Find the library :)
    FIND_LIBRARY(Teem_LIBRARIES teem ${Teem_LIBRARY_DIRS})
    #    LINK_DIRECTORIES   (${TEEM_LIBRARY_DIRS})

  ELSE (FOUND_TEEMCONFIG_CMAKE)
    # Warn about version.
    MESSAGE("TEEMConfig.cmake not found. Are you using Teem 1.10 source distribution?")
  ENDIF (FOUND_TEEMCONFIG_CMAKE)

  MARK_AS_ADVANCED(FOUND_TEEMCONFIG_CMAKE)
  MARK_AS_ADVANCED(FOUND_TEEM_BIN)

ENDIF(FOUND_TEEM_BIN)




