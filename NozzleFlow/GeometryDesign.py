import numpy as np

def cooling_geometry(dic: dict):
    """Generates the allowable number of cooling channels"""
    thickness_wall = dic["W"]["thickness"]
    throat_radius = dic["E"]["r_throat"]
    thickness_fin = dic["C"]["spacing"]

    N = int(np.floor(2*np.pi*(throat_radius + thickness_wall) / (2*thickness_fin)))

    # if N > dic["C"]["num_ch"]:
    dic["C"]["num_ch"] = N

