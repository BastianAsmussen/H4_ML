{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = {
    self,
    nixpkgs,
  }: let
    supportedSystems = ["x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin"];
    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    pkgs = forAllSystems (system: nixpkgs.legacyPackages.${system});
  in {
    devShells = forAllSystems (system: {
      default = pkgs.${system}.mkShellNoCC {
        packages = with pkgs.${system}; [
          (python3.withPackages (pythonPkgs:
            with pythonPkgs; [
              pandas
              matplotlib
              openpyxl
              nltk
              wordcloud
              scikit-learn
              tensorflow
              keras
              seaborn
              imbalanced-learn
              pillow
              opencv-python
              numpy
              xgboost
            ]))
        ];
      };
    });
  };
}
