{ lib
, buildPythonPackage
, fetchPypi
, fetchFromGitHub
, pkg-config
, cmake
, stdenv
, gperftools # better performance according to the docs
}:

let
 version = "0.1.84";
 src = fetchFromGitHub {
        # unfortunately the pypi distribution is broken due to VERSION (but we have to build the C++ lib so we need this source anyway)
        owner = "google";
        repo = "sentencepiece";
        rev = "v${version}";
        sha256 = "144y25nj4rwxmgvzqbr7al9fjwh3539ssjswvzrx4gsgfk62lsm0";
      };

 cppPkg = stdenv.mkDerivation rec { # need to build the C++ lib
      pname = "sentencepiece-cpp";
      inherit version src;
      buildInputs  = [ cmake ];
      propagatedBuildInputs = [ gperftools ];
      meta = {
      };
    };
in buildPythonPackage rec {
      pname = "sentencepiece";
      inherit version src;
      sourceRoot = "source/python";
      nativeBuildInputs = [ pkg-config ] ;
      propagatedBuildInputs = [ cppPkg ]; # depend on the C++ lib
      doCheck = false;
      meta = {
      };
    }
