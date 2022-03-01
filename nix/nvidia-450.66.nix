nixpkgs_source: self: pkgs:
         with pkgs; {
           linuxPackages = pkgs.linuxPackages.extend (self: super: {
              nvidia_x11 = callPackage (import (nixpkgs_source + "/pkgs/os-specific/linux/nvidia-x11/generic.nix") {
                version = "450.66";
                sha256_64bit = "1a6va0gvbzpkyza693v2ml1is4xbv8wxasqk0zd5y7rxin94c1ms";
                settingsSha256 = "1677g7rcjbcs5fja1s4p0syhhz46g9x2qqzyn3wwwrjsj7rwaz78";
                persistencedSha256 = "01kvd3zp056i4n8vazj7gx1xw0h4yjdlpazmspnsmwg24ijb83x4";
              }) {
                libsOnly = true;
              };
            });
          }
