{ bootstrap ? import <nixpkgs> {} }:

let
    pkgs_source = fetchTarball "https://github.com/NixOS/nixpkgs/archive/dfa8e8b9bc4a18bab8f2c55897b6054d31e2c71b.tar.gz";
    overlays = [
      (self: super:  # define our local packages
         {
          python3 = super.python3.override {
           packageOverrides = python-self: python-super: {
	     sacremoses = python-self.callPackage /opt/nix/sacremoses-0.0.35.nix {};
	     sentencepiece = python-self.callPackage /opt/nix/sentencepiece-0.1.84.nix {};
             transformers = python-self.callPackage /opt/nix/transformers-2.1.1.nix { };
           };};})
      ((import /opt/nix/nvidia-450.66.nix  ) pkgs_source )  # fix version of nvidia drivers
      (self: super: {
          cudatoolkit = super.cudatoolkit_10; # fix version of cuda
          cudnn = super.cudnn_cudatoolkit_10;})
    ];
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
    pkgs = import pkgs_source {inherit overlays; inherit config;};
    py = pkgs.python3;
    pyEnv = py.buildEnv.override {
      extraLibs = with py.pkgs;
        [
         # If you want to have a local virtualenv, see here: https://github.com/NixOS/nixpkgs/blob/master/doc/languages-frameworks/python.section.md
         transformers 
         pytorch
	 nltk
         notebook
         matplotlib
         plotly
         pandas
         scikitlearn
         # notebook
         # etc
        ];
      ignoreCollisions = true;};
in
  pkgs.stdenv.mkDerivation {
    name = "sh-env";
    buildInputs = [pyEnv pkgs.ranger pkgs.htop];
    shellHook = ''
      export LANG=en_US.UTF-8
      export PYTHONIOENCODING=UTF-8
      export LD_PRELOAD=/lib64/libcuda.so.1
     '';
  }
