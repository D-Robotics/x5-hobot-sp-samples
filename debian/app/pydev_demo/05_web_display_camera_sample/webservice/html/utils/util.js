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

// 获取 url query 参数
function getUrlQueryParameter (name) {
  var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)", "i");
  var r = window.location.search.substr(1).match(reg);
  if (r != null) return decodeURI(r[2]);
  return null;
}

// 生成随机数
function randomNum(minNum, maxNum) {
  switch (arguments.length) {
    case 1:
      return parseInt(Math.random() * minNum + 1, 10);
      break;
    case 2:
      return parseInt(Math.random() * (maxNum - minNum + 1) + minNum, 10);
      break;
    default:
      return 0;
      break;
  }
}

function toNormalTime(timestamp) {
  var date = new Date(timestamp),
    hour = date.getHours() + ":",
    minutes = date.getMinutes(),
    seconds = ":" + date.getSeconds();

  return hour + minutes + seconds;
}

// 视频保存数据相关
/**
 * 开始保存  保存名字
 * */
function save_file_start(name) {
  var file_name = name || new Date().valueOf();
  $.ajax({
    type: "POST",
    url: "/save_file_start",
    data: {
      name: file_name
    },
    dataType: "json",
    success: function(data) {
      console.log(data);
      _issavedata = true;
    },
    error: function(json) {
      console.log("error");
    }
  });
}
/**
 * 添加数据
 * */
function save_file_data(imgdata) {
  $.ajax({
    type: "POST",
    url: "/save_file_data",
    data: imgdata,
    dataType: "json",
    success: function(data) {
      console.log(data);
    },
    error: function(json) {
      console.log("error");
    }
  });
}
/**
 * 保存数据
 * */
function save_file_json() {
  isimg = false;
  _issavedata = false;
  $.ajax({
    type: "POST",
    url: "/save_file_json",
    dataType: "json",
    success: function(data) {
      console.log(data);
    },
    error: function(json) {
      console.log("error");
    }
  });
}
