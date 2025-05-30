// Copyright (c) 2024，D-Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import inherits from 'inherits';

// var inherits = require('inherits');

var getNormals = require('polyline-normals');

var VERTS_PER_POINT = 2;

export default function createLineMesh (THREE) {

    function LineMesh (path, opt) {

        if (!(this instanceof LineMesh)) {

            return new LineMesh(path, opt);
    
        }
        THREE.BufferGeometry.call(this);

        if (Array.isArray(path)) {

            opt = opt || {};
    
        } else if (typeof path === 'object') {

            opt = path;
            path = [];
    
        }

        opt = opt || {};

        this.addAttribute('position', new THREE.BufferAttribute(undefined, 3));
        this.addAttribute('lineNormal', new THREE.BufferAttribute(undefined, 2));
        this.addAttribute('lineMiter', new THREE.BufferAttribute(undefined, 1));
        if (opt.distances) {

            this.addAttribute('lineDistance', new THREE.BufferAttribute(undefined, 1));
    
        }
        if (typeof this.setIndex === 'function') {

            this.setIndex(new THREE.BufferAttribute(undefined, 1));
    
        } else {

            this.addAttribute('index', new THREE.BufferAttribute(undefined, 1));
    
        }
        this.update(path, opt.closed);
  
    }

    inherits(LineMesh, THREE.BufferGeometry);

    LineMesh.prototype.update = function (path, closed) {

        path = path || [];
        var normals = getNormals(path, closed);

        if (closed) {

            path = path.slice();
            path.push(path[0]);
            normals.push(normals[0]);
    
        }

        var attrPosition = this.getAttribute('position');
        var attrNormal = this.getAttribute('lineNormal');
        var attrMiter = this.getAttribute('lineMiter');
        var attrDistance = this.getAttribute('lineDistance');
        var attrIndex = typeof this.getIndex === 'function' ? this.getIndex() : this.getAttribute('index');

        var indexCount = Math.max(0, (path.length - 1) * 6);
        if (!attrPosition.array ||
        (path.length !== attrPosition.array.length / 3 / VERTS_PER_POINT)) {

            var count = path.length * VERTS_PER_POINT;
            attrPosition.array = new Float32Array(count * 3);
            attrNormal.array = new Float32Array(count * 2);
            attrMiter.array = new Float32Array(count);
            attrIndex.array = new Uint16Array(indexCount);

            if (attrDistance) {

                attrDistance.array = new Float32Array(count);
      
            }
    
        }

        if (undefined !== attrPosition.count) {

            attrPosition.count = count;
    
        }
        attrPosition.needsUpdate = true;

        if (undefined !== attrNormal.count) {

            attrNormal.count = count;
    
        }
        attrNormal.needsUpdate = true;

        if (undefined !== attrMiter.count) {

            attrMiter.count = count;
    
        }
        attrMiter.needsUpdate = true;

        if (undefined !== attrIndex.count) {

            attrIndex.count = indexCount;
    
        }
        attrIndex.needsUpdate = true;

        if (attrDistance) {

            if (undefined !== attrDistance.count) {

                attrDistance.count = count;
      
            }
            attrDistance.needsUpdate = true;
    
        }

        var index = 0;
        var c = 0;
        var dIndex = 0;
        var indexArray = attrIndex.array;

        path.forEach(function (point, pointIndex, list) {

            var i = index;
            indexArray[c++] = i + 0;
            indexArray[c++] = i + 1;
            indexArray[c++] = i + 2;
            indexArray[c++] = i + 2;
            indexArray[c++] = i + 1;
            indexArray[c++] = i + 3;

            attrPosition.setXYZ(index++, point[0], point[1], point[2]);
            attrPosition.setXYZ(index++, point[0], point[1], point[2]);

            if (attrDistance) {

                var d = pointIndex / (list.length - 1);
                attrDistance.setX(dIndex++, d);
                attrDistance.setX(dIndex++, d);
      
            }
    
        });

        var nIndex = 0;
        var mIndex = 0;
        normals.forEach(function (n) {

            var norm = n[0];
            var miter = n[1];
            attrNormal.setXY(nIndex++, norm[0], norm[1]);
            attrNormal.setXY(nIndex++, norm[0], norm[1]);

            attrMiter.setX(mIndex++, -miter);
            attrMiter.setX(mIndex++, miter);
    
        });
  
    };

    return LineMesh;

};
