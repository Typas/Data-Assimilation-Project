using Pkg;
Pkg.add("OrdinaryDiffEq")
Pkg.add("Optim")
Pkg.add("PackageCompiler")

using PackageCompiler;
PackageCompiler.create_sysimage([:OrdinaryDiffEq, :Optim]; sysimage_path="da-julia.so")
