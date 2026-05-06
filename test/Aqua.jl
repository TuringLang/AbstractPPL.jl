module AquaTests

using Aqua: Aqua
using AbstractPPL

# `persistent_tasks` spawns a subprocess that runs `Pkg.precompile()` on a
# wrapper package depending on AbstractPPL. On Julia 1.10, this hits
# "Declaring __precompile__(false) is not allowed in files that are being
# precompiled" inside the wrapper's extension precompile path — a known
# brittleness in the Aqua/Julia 1.10 interaction, not in our extensions
# (the dedicated `Ext` CI jobs load and exercise every extension on `min`
# Julia and pass). Re-enable once `min` is bumped past 1.10.
Aqua.test_all(AbstractPPL; persistent_tasks=VERSION >= v"1.11")

end
