"""MOSDAC adapter (port target, stub). INSAT-3D/3DR IR1 10.8um ~ GOES C13; DWR gives
reflectivity + radial velocity (enables downburst). Needs MOSDAC SSO + order/API access.
Must return the same event schema as sevir.py: {vil, ir, lght, id, source} on the 1 km/5-min grid."""


def load_insat_event(path):
    raise NotImplementedError("Pending MOSDAC access - see CLAUDE.md section 3.2")


def load_dwr_volume(path):
    raise NotImplementedError("Pending MOSDAC DWR access - radial velocity for downburst (roadmap)")
