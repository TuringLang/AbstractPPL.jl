module AquaTests

using Aqua: Aqua
using AbstractPPL

# For now, we skip ambiguities since they come from interactions
# with third-party packages rather than issues in AbstractPPL itself
Aqua.test_all(AbstractPPL; ambiguities=false)

end