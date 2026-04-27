module AquaTests

using Aqua: Aqua
using AbstractPPL

Aqua.test_all(AbstractPPL; stale_deps=(; ignore=[:ADTypes]))

end
