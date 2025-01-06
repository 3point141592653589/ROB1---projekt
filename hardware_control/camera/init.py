from basler_camera import BaslerCamera


def init_camera():
    camera: BaslerCamera = BaslerCamera()

    camera.connect_by_name("camera-crs97")

    camera.open()
    camera.set_parameters()
    camera.start()
    return camera
