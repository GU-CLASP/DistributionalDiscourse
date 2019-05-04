{ nixpkgs ? import <nixpkgs> {} }:
let
   nixpkgs_source = fetchTarball https://github.com/NixOS/nixpkgs/archive/19.03.tar.gz;
   # Older configuration:
   # nixpkgs_source = nixpkgs.fetchFromGitHub { # for safety of checking the hash
   #   owner = "jyp";
   #   repo = "nixpkgs";
   #   rev = "cudnn7.3-cuda9.0";
   #   sha256 = "1jvsagry3842slgzxkqmp66mxs5b3clbpm7xijv1vjp2wxxvispf";
   # };
   
in
with (import nixpkgs_source {}).pkgs;

let py = (python36.buildEnv.override {
  extraLibs =  with python36Packages; 
    [nltk
     pytorch
     notebook
     # you can add other packages here. (comment out as necessary)
     # scikitimage
     # gensim
     # scikitlearn
    ];
  ignoreCollisions = true;});
in pkgs.stdenv.mkDerivation {
    name = "python-env";
    buildInputs = [ py ];
    shellHook = ''
      export PYTHONIOENCODING=UTF-8
      '';
     }
