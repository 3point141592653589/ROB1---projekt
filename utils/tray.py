TRAY_MARKER_SIZE = 0.036
TRAY_ID2_CENTER = (0.18, 0.14)
TRAY_CORNERS = [[-0.03, -0.03, 0], [0.21, -0.03, 0], [0.21, 0.17, 0], [-0.03, 0.17, 0]]


def get_tray_dict(id_origin, id2):
    return {
        id_origin: ((0, 0), TRAY_MARKER_SIZE),
        id2: (TRAY_ID2_CENTER, TRAY_MARKER_SIZE),
    }
