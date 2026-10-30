"""
Rendering of the book's MuJoCo runs as videos and stills.

Physical AI, Together, Chapters 13 to 15 (the videos of their governed runs).

The chapters' video scripts run their configurations in MuJoCo again with
the book's modules unchanged, record the simulated state (shapes.Recorder)
and then draw it. Nothing here takes part in the physics: the drawing uses
a copy of each robot's model to which only visual elements are added (no
mass, no contact), with its state set from the record.

  egl.py       a headless OpenGL context through EGL, with ctypes only
  film.py      the renderer (MuJoCo's own, through MjrContext), supersampled
               and anti-aliased; annotations, a colour grade, captions, a
               camera that follows a target smoothly, and an H.264 writer
               through ffmpeg
  stage.py     the sky, the sun and the fill light, haze and shadows
  shoot.py     the frames drawn in parallel and joined; stills
  shapes.py    the recorder, and the scenery's shapes
  textures.py  procedural textures from fixed seeds
  collada.py   a reader of COLLADA meshes
  urdf.py      a URDF's visual elements on the imported model's bodies
  export.py    a video's scene and frames written for a path tracer
  cycles/      the path tracer's side, Blender's Cycles on a GPU (run by
               the reader: README.md there)

MuJoCo's renderer rasterizes with OpenGL: directional lights, a shadow map,
a mirror reflection on one flat surface at most, a gradient sky and haze,
but no global illumination or ray tracing, so the images are engineering
renders of simulated states, not photographs; cycles/ path traces the same
scenes and poses.
"""
