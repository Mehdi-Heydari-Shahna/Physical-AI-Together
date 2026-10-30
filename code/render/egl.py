"""
A headless OpenGL context through EGL, with ctypes only.

Physical AI, Together, Chapters 13 to 15 (the videos of their governed runs).

MuJoCo's Python renderer creates its context with PyOpenGL. This module
needs only libEGL: it opens Mesa's surfaceless platform if it is offered,
or a GPU's device platform, or the default display, and makes a context
current on a small pbuffer. MuJoCo then renders into its own offscreen
framebuffer (MjrContext with mjFB_OFFSCREEN).
"""
import ctypes

EGL_SURFACE_TYPE, EGL_PBUFFER_BIT = 0x3033, 0x0001
EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_BLUE_SIZE, EGL_ALPHA_SIZE = 0x3024, 0x3023, 0x3022, 0x3021
EGL_DEPTH_SIZE, EGL_STENCIL_SIZE = 0x3025, 0x3026
EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE = 0x3040, 0x0008, 0x3038
EGL_OPENGL_API = 0x30A2
EGL_WIDTH, EGL_HEIGHT = 0x3057, 0x3056
EGL_VENDOR, EGL_VERSION = 0x3053, 0x3054
EGL_PLATFORM_SURFACELESS_MESA = 0x31DD
EGL_PLATFORM_DEVICE_EXT = 0x313F

_p = ctypes.c_void_p


def _library():
    for name in ("libEGL.so.1", "libEGL.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("libEGL is not installed: the videos need EGL (Mesa or a GPU driver)")


class Context:
    """An OpenGL context made current in this thread for as long as the
    object lives."""

    def __init__(self):
        egl = self.egl = _library()
        egl.eglGetProcAddress.restype = _p
        egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        egl.eglGetDisplay.restype = _p
        egl.eglGetDisplay.argtypes = [_p]
        egl.eglInitialize.argtypes = [_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        egl.eglChooseConfig.argtypes = [_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(_p),
                                        ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        egl.eglBindAPI.argtypes = [ctypes.c_uint]
        egl.eglCreateContext.restype = _p
        egl.eglCreateContext.argtypes = [_p, _p, _p, ctypes.POINTER(ctypes.c_int)]
        egl.eglCreatePbufferSurface.restype = _p
        egl.eglCreatePbufferSurface.argtypes = [_p, _p, ctypes.POINTER(ctypes.c_int)]
        egl.eglMakeCurrent.argtypes = [_p, _p, _p, _p]
        egl.eglQueryString.restype = ctypes.c_char_p
        egl.eglQueryString.argtypes = [_p, ctypes.c_int]
        egl.eglGetError.restype = ctypes.c_int
        egl.eglDestroyContext.argtypes = [_p, _p]
        egl.eglTerminate.argtypes = [_p]

        self.display = None
        for display in self._displays():
            major, minor = ctypes.c_int(), ctypes.c_int()
            if display and egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
                self.display = display
                break
        if not self.display:
            raise RuntimeError(f"EGL could not open a display (error {egl.eglGetError():#x})")
        attrs = (ctypes.c_int * 15)(EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RED_SIZE, 8,
                                    EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                                    EGL_DEPTH_SIZE, 24, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                                    EGL_NONE)
        config, n = _p(), ctypes.c_int()
        if not egl.eglChooseConfig(self.display, attrs, ctypes.byref(config), 1, ctypes.byref(n)) \
                or n.value < 1:
            raise RuntimeError(f"EGL offers no OpenGL configuration (error {egl.eglGetError():#x})")
        egl.eglBindAPI(EGL_OPENGL_API)
        self.context = egl.eglCreateContext(self.display, config, None, None)
        if not self.context:
            raise RuntimeError(f"EGL could not create a context (error {egl.eglGetError():#x})")
        size = (ctypes.c_int * 5)(EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE)
        self.surface = egl.eglCreatePbufferSurface(self.display, config, size)
        if not egl.eglMakeCurrent(self.display, self.surface, self.surface, self.context):
            raise RuntimeError(f"EGL could not make the context current "
                               f"(error {egl.eglGetError():#x})")
        self.vendor = (egl.eglQueryString(self.display, EGL_VENDOR) or b"").decode()
        self.version = (egl.eglQueryString(self.display, EGL_VERSION) or b"").decode()

    def _displays(self):
        egl = self.egl
        address = egl.eglGetProcAddress(b"eglGetPlatformDisplayEXT")
        if address:
            platform = ctypes.CFUNCTYPE(_p, ctypes.c_uint, _p, _p)(address)
            yield platform(EGL_PLATFORM_SURFACELESS_MESA, None, None)
            query = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
            if query:
                devices = (_p * 8)()
                n = ctypes.c_int()
                ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.POINTER(_p),
                                 ctypes.POINTER(ctypes.c_int))(query)(8, devices, ctypes.byref(n))
                for i in range(n.value):
                    yield platform(EGL_PLATFORM_DEVICE_EXT, devices[i], None)
        yield egl.eglGetDisplay(None)

    def close(self):
        if self.display:
            self.egl.eglMakeCurrent(self.display, None, None, None)
            self.egl.eglDestroyContext(self.display, self.context)
            self.egl.eglTerminate(self.display)
            self.display = None
