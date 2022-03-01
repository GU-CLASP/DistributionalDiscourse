{ lib
, buildPythonPackage
, fetchPypi
, click
, six
, tqdm
, joblib
}:
buildPythonPackage rec {
  pname = "sacremoses";
  version = "0.0.35";
  src = fetchPypi {
    inherit pname version;
    sha256 = "02j11r0lkr192669xdvif1sxki9p8xdfn3wc2jy8pz6vrfaxm10y";
  };
  propagatedBuildInputs = [ click six joblib tqdm ];
  checkInputs = [  ];
  doCheck = false;
  meta = {
    description = "TODO";
    homepage = https://bert.ai.com;
    license = lib.licenses.lgpl21;
    maintainers = with lib.maintainers; [ ];
  };
}
