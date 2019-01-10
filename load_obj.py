"""
Adapted from https://github.com/cprogrammer1994/ModernGL.ext.obj
"""

import logging
import re
import struct
import numpy as np
from typing import Tuple

log = logging.getLogger('load_obj')

RE_COMMENT = re.compile(r'#[^\n]*\n', flags=re.M)
RE_VERT_COLOR = re.compile(r'^v\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)$')

RE_VERT = re.compile(r'^v\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)$')

RE_TEXT = re.compile(r'^vt\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)(?:\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?))?$')
RE_NORM = re.compile(r'^vn\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[Ee]-?\d+)?)$')
RE_FACE = re.compile(r'^f\s+(\d+)(/(\d+)?(/(\d+))?)?\s+(\d+)(/(\d+)?(/(\d+))?)?\s+(\d+)(/(\d+)?(/(\d+))?)?$')

PACKER = 'lambda vx, vy, vz, tx, ty, tz, nx, ny, nz, cr, cg, cb: struct.pack("%df", %s)'

def default_packer(vx, vy, vz, tx, ty, tz, nx, ny, nz, cr, cg, cb):
    return struct.pack('9f', vx, vy, vz, tx, ty, tz, nx, ny, nz, cr, cg, cb)

def int_or_none(x):
    return None if x is None else int(x)

def safe_float(x):
    return 0.0 if x is None else float(x)

class Obj:
    '''
    Wavefront .obj file
    '''

    @staticmethod
    def open(filename, has_vertex_coloring=False) -> 'Obj':
        '''
            Args:
                filename (str): The filename.
            Returns:
                Obj: The object.
            Examples:
                .. code-block:: python
                    import ModernGL
                    from ModernGL.ext import obj
                    model = obj.Obj.open('box.obj')
        '''

        return Obj.fromstring(open(filename).read(), has_vertex_coloring=has_vertex_coloring)

    @staticmethod
    def open_as_np_array(filename, has_vertex_coloring=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Args:
                filename (str): The filename.
        Returns:
                 Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                 tuple of vert, text, norm, face, color
        Examples:
            .. code-block:: python
                import ModernGL
                from ModernGL.ext import obj
                vert, text, norm, face, color = obj.Obj.open_as_np_array('box.obj')
        '''
        return Obj.fromstring(open(filename).read(), return_np_array=True)

    @staticmethod
    def frombytes(data) -> 'Obj':
        '''
            Args:
                data (bytes): The obj file content.
            Returns:
                Obj: The object.
            Examples:
                .. code-block:: python
                    import ModernGL
                    from ModernGL.ext import obj
                    content = open('box.obj', 'rb').read()
                    model = obj.Obj.frombytes(content)
        '''

        return Obj.fromstring(data.decode())

    @staticmethod
    def fromstring(data, return_np_array=False, has_vertex_coloring=False) -> 'Obj':
        '''
            Args:
                data (str): The obj file content.
            Returns:
                Obj: The object.
            Examples:
                .. code-block:: python
                    import ModernGL
                    from ModernGL.ext import obj
                    content = open('box.obj').read()
                    model = obj.Obj.fromstring(content)
        '''

        vert = []
        text = []
        norm = []
        face_vtn = []
        face = []
        color = []
        '''
        face_vtn is [vertex texture normal]
        face is [vertex1, vertex2, vertex3]
        '''


        data = RE_COMMENT.sub('\n', data)

        for line in data.splitlines():
            line = line.strip()

            if not line:
                continue
            if has_vertex_coloring:
                match = RE_VERT_COLOR.match(line)
            else:
                match = RE_VERT.match(line)

            if match:
                _l = tuple(map(safe_float, match.groups()))
                vert.append(_l[:3])
                if has_vertex_coloring:
                    color.append(_l[3:])
                    continue

            match = RE_TEXT.match(line)
            if match:
                text.append(tuple(map(safe_float, match.groups())))
                continue

            match = RE_NORM.match(line)

            if match:
                norm.append(tuple(map(safe_float, match.groups())))
                continue

            match = RE_FACE.match(line)

            if match:
                v, t, n = match.group(1, 3, 5)
                face_vtn.append((int(v), int_or_none(t), int_or_none(n)))
                v, t, n = match.group(6, 8, 10)
                face_vtn.append((int(v), int_or_none(t), int_or_none(n)))
                v, t, n = match.group(11, 13, 15)
                face_vtn.append((int(v), int_or_none(t), int_or_none(n)))
                v1, v2, v3 = match.group(1, 6, 11)
                face.append( (int(v1), int(v2), int(v3)) )
                continue

            log.debug('unknown line "%s"', line)

        if not face_vtn:
            raise Exception('empty')

        t0, n0 = face_vtn[0][1:3]

        for v, t, n in face_vtn:
            #if (t0 is None) ^ (t is None):
                #raise Exception('inconsinstent texture coords')

            if (n0 is None) ^ (n is None):
                raise Exception('inconsinstent normals')
        print (' fromstring ', len(vert), len(text), len(norm), len(face_vtn), len(face), len(color) )
        if return_np_array:
            return np.asarray(vert), np.asarray(text), np.asarray(norm), np.asarray(face_vtn), np.asarray(face), np.asarray(color)
        else:
            return Obj(vert, text, norm, face_vtn, color)

    def __init__(self, vert, text, norm, face, color):
        self.vert = vert
        self.text = text
        self.norm = norm
        self.face = face
        self.color = color

    def pack(self, packer=default_packer) -> bytes:
        '''
            Args:
                packer (str or lambda): The vertex attributes to pack.
            Returns:
                bytes: The packed vertex data.
            Examples:
                .. code-block:: python
                    import ModernGL
                    from ModernGL.ext import obj
                    model = obj.Obj.open('box.obj')
                    # default packer
                    data = model.pack()
                    # same as the default packer
                    data = model.pack('vx vy vz tx ty tz nx ny nz')
                    # pack vertices
                    data = model.pack('vx vy vz')
                    # pack vertices and texture coordinates (xy)
                    data = model.pack('vx vy vz tx ty')
                    # pack vertices and normals
                    data = model.pack('vx vy vz nx ny nz')
                    # pack vertices with padding
                    data = model.pack('vx vy vz 0.0')
        '''

        if isinstance(packer, str):
            nodes = packer.split()
            packer = eval(PACKER % (len(nodes), ', '.join(nodes)))

        result = bytearray()

        for v, t, n in self.face:
            vx, vy, vz = self.vert[v - 1]
            cr, cg, cb = self.color[v - 1]
            tx, ty, tz = self.text[t - 1] if t is not None else (0.0, 0.0, 0.0)
            nx, ny, nz = self.norm[n - 1] if n is not None else (0.0, 0.0, 0.0)
            result += packer(vx, vy, vz, tx, ty, tz, nx, ny, nz, cr, cg, cb)

        return bytes(result)

