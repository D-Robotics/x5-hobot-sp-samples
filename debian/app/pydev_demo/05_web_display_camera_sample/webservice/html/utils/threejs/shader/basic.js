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

var assign = require('object-assign');

export default function (THREE) {

    return function (opt) {

        opt = opt || {};
        var thickness = typeof opt.thickness === 'number' ? opt.thickness : 0.1;
        var opacity = typeof opt.opacity === 'number' ? opt.opacity : 1.0;
        var diffuse = opt.diffuse !== null ? opt.diffuse : 0xffffff;

    // remove to satisfy r73
        delete opt.thickness;
        delete opt.opacity;
        delete opt.diffuse;
        delete opt.precision;

        var ret = assign({
            uniforms: {
                thickness: { type: 'f', value: thickness },
                opacity: { type: 'f', value: opacity },
                diffuse: { type: 'c', value: new THREE.Color(diffuse) }
            },
            vertexShader: [
                'uniform float thickness;',
                'attribute float lineMiter;',
                'attribute vec2 lineNormal;',
                'varying vec3 vPosition;',
                'void main() {',
                'vPosition = position;',
                'vec3 pointPos = position.xyz + vec3(lineNormal * thickness / 2.0 * lineMiter, 0.0);',
                'gl_Position = projectionMatrix * modelViewMatrix * vec4(pointPos, 1.0);',
                '}'
            ].join('\n'),
            fragmentShader: [
                'uniform vec3 diffuse;',
                'uniform float opacity;',
                'void main() {',
                'gl_FragColor = vec4(diffuse, opacity);',
                '}'
            ].join('\n'),
            transparent: true,             //增加透明度控制
            side: THREE.DoubleSide
        }, opt);

        var threeVers = (parseInt(THREE.REVISION, 10) || 0) | 0;
        if (threeVers < 72) {

      // Old versions need to specify shader attributes
            ret.attributes = {
                lineMiter: { type: 'f', value: 0 },
                lineNormal: { type: 'v2', value: new THREE.Vector2() }
            };
    
        }
        return ret;
  
    };

};
