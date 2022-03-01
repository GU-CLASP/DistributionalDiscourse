{ stdenv , buildPythonPackage , fetchPypi
, pytorch , numpy , boto3 , requests , tqdm , regex , sacremoses, sentencepiece 
}:
buildPythonPackage rec {
  pname = "transformers";
  version = "2.1.1";

  src = fetchPypi {
    inherit pname version;
    sha256 = "1gdp0bx11ara78wlpv12wmvh345pnq04qgni3b753vi9168fw2mw";
  };

  checkInputs = [  ];
  propagatedBuildInputs = [ pytorch numpy boto3 requests tqdm regex sacremoses sentencepiece];
  doCheck = false;

  meta = with stdenv.lib; {
    description = "TODO";
    homepage = https://bert.ai.com;
    license = licenses.lgpl21;
  };
}
