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

/* eslint-disable */
// https://github.com/joewalnes/reconnecting-websocket
;(function(global, factory) {
    if (typeof define === 'function' && define.amd) {
      define([], factory)
    } else if (typeof module !== 'undefined' && module.exports) {
      module.exports = factory()
    } else {
      global.ReconnectingWebSocket = factory()
    }
  })(this, function() {
    if (!('WebSocket' in window)) {
      return
    }
  
    function ReconnectingWebSocket(url, protocols, options) {
      // Default settings
      var settings = {
        /** Whether this instance should log debug messages. */
        debug: false,
  
        /** Whether or not the websocket should attempt to connect immediately upon instantiation. */
        automaticOpen: true,
  
        /** The number of milliseconds to delay before attempting to reconnect. */
        reconnectInterval: 1000,
        /** The maximum number of milliseconds to delay a reconnection attempt. */
        maxReconnectInterval: 30000,
        /** The rate of increase of the reconnect delay. Allows reconnect attempts to back off when problems persist. */
        reconnectDecay: 1.5,
  
        /** The maximum time in milliseconds to wait for a connection to succeed before closing and retrying. */
        timeoutInterval: 2000,
  
        /** The maximum number of reconnection attempts to make. Unlimited if null. */
        maxReconnectAttempts: null,
  
        /** The binary type, possible values 'blob' or 'arraybuffer', default 'blob'. */
        binaryType: 'blob'
      }
      if (!options) {
        options = {}
      }
  
      // Overwrite and define settings with options if they exist.
      for (var key in settings) {
        if (typeof options[key] !== 'undefined') {
          this[key] = options[key]
        } else {
          this[key] = settings[key]
        }
      }
  
      // These should be treated as read-only properties
  
      /** The URL as resolved by the constructor. This is always an absolute URL. Read only. */
      this.url = url
  
      /** The number of attempted reconnects since starting, or the last successful connection. Read only. */
      this.reconnectAttempts = 0
  
      /**
       * The current state of the connection.
       * Can be one of: WebSocket.CONNECTING, WebSocket.OPEN, WebSocket.CLOSING, WebSocket.CLOSED
       * Read only.
       */
      this.readyState = WebSocket.CONNECTING
  
      /**
       * A string indicating the name of the sub-protocol the server selected; this will be one of
       * the strings specified in the protocols parameter when creating the WebSocket object.
       * Read only.
       */
      this.protocol = null
  
      // Private state variables
  
      var self = this
      var ws
      var forcedClose = false
      var timedOut = false
      var eventTarget = document.createElement('div')
  
      // Wire up "on*" properties as event handlers
  
      eventTarget.addEventListener('open', function(event) {
        self.onopen(event)
      })
      eventTarget.addEventListener('close', function(event) {
        self.onclose(event)
      })
      eventTarget.addEventListener('connecting', function(event) {
        self.onconnecting(event)
      })
      eventTarget.addEventListener('message', function(event) {
        self.onmessage(event)
      })
      eventTarget.addEventListener('error', function(event) {
        self.onerror(event)
      })
  
      // Expose the API required by EventTarget
  
      this.addEventListener = eventTarget.addEventListener.bind(eventTarget)
      this.removeEventListener = eventTarget.removeEventListener.bind(eventTarget)
      this.dispatchEvent = eventTarget.dispatchEvent.bind(eventTarget)
  
      /**
       * This function generates an event that is compatible with standard
       * compliant browsers and IE9 - IE11
       *
       * This will prevent the error:
       * Object doesn't support this action
       *
       * http://stackoverflow.com/questions/19345392/why-arent-my-parameters-getting-passed-through-to-a-dispatched-event/19345563#19345563
       * @param s String The name that the event should use
       * @param args Object an optional object that the event will use
       */
      function generateEvent(s, args) {
        var evt = document.createEvent('CustomEvent')
        evt.initCustomEvent(s, false, false, args)
        return evt
      }
  
      this.open = function(reconnectAttempt) {
        ws = new WebSocket(self.url, protocols || [])
        ws.binaryType = this.binaryType
  
        if (reconnectAttempt) {
          if (
            this.maxReconnectAttempts &&
            this.reconnectAttempts > this.maxReconnectAttempts
          ) {
            return
          }
        } else {
          eventTarget.dispatchEvent(generateEvent('connecting'))
          this.reconnectAttempts = 0
        }
  
        if (self.debug || ReconnectingWebSocket.debugAll) {
          console.debug('ReconnectingWebSocket', 'attempt-connect', self.url)
        }
  
        var localWs = ws
        var timeout = setTimeout(function() {
          if (self.debug || ReconnectingWebSocket.debugAll) {
            console.debug('ReconnectingWebSocket', 'connection-timeout', self.url)
          }
          timedOut = true
          localWs.close()
          timedOut = false
        }, self.timeoutInterval)
  
        ws.onopen = function(event) {
          clearTimeout(timeout)
          if (self.debug || ReconnectingWebSocket.debugAll) {
            console.debug('ReconnectingWebSocket', 'onopen', self.url)
          }
          self.protocol = ws.protocol
          self.readyState = WebSocket.OPEN
          self.reconnectAttempts = 0
          var e = generateEvent('open')
          e.isReconnect = reconnectAttempt
          reconnectAttempt = false
          eventTarget.dispatchEvent(e)
        }
  
        ws.onclose = function(event) {
          clearTimeout(timeout)
          ws = null
          if (forcedClose) {
            self.readyState = WebSocket.CLOSED
            eventTarget.dispatchEvent(generateEvent('close'))
          } else {
            self.readyState = WebSocket.CONNECTING
            var e = generateEvent('connecting')
            e.code = event.code
            e.reason = event.reason
            e.wasClean = event.wasClean
            eventTarget.dispatchEvent(e)
            if (!reconnectAttempt && !timedOut) {
              if (self.debug || ReconnectingWebSocket.debugAll) {
                console.debug('ReconnectingWebSocket', 'onclose', self.url)
              }
              eventTarget.dispatchEvent(generateEvent('close'))
            }
  
            var timeout =
              self.reconnectInterval *
              Math.pow(self.reconnectDecay, self.reconnectAttempts)
            setTimeout(function() {
              self.reconnectAttempts++
              self.open(true)
            }, timeout > self.maxReconnectInterval
              ? self.maxReconnectInterval
              : timeout)
          }
        }
        ws.onmessage = function(event) {
          if (self.debug || ReconnectingWebSocket.debugAll) {
            console.debug(
              'ReconnectingWebSocket',
              'onmessage',
              self.url,
              event.data
            )
          }
          var e = generateEvent('message')
          e.data = event.data
          eventTarget.dispatchEvent(e)
        }
        ws.onerror = function(event) {
          if (self.debug || ReconnectingWebSocket.debugAll) {
            console.debug('ReconnectingWebSocket', 'onerror', self.url, event)
          }
          eventTarget.dispatchEvent(generateEvent('error'))
        }
      }
  
      // Whether or not to create a websocket upon instantiation
      if (this.automaticOpen == true) {
        this.open(false)
      }
  
      /**
       * Transmits data to the server over the WebSocket connection.
       *
       * @param data a text string, ArrayBuffer or Blob to send to the server.
       */
      this.send = function(data) {
        if (ws) {
          if (self.debug || ReconnectingWebSocket.debugAll) {
            console.debug('ReconnectingWebSocket', 'send', self.url, data)
          }
          return ws.send(data)
        } else {
          throw 'INVALID_STATE_ERR : Pausing to reconnect websocket'
        }
      }
  
      /**
       * Closes the WebSocket connection or connection attempt, if any.
       * If the connection is already CLOSED, this method does nothing.
       */
      this.close = function(code, reason) {
        // Default CLOSE_NORMAL code
        if (typeof code == 'undefined') {
          code = 1000
        }
        forcedClose = true
        if (ws) {
          ws.close(code, reason)
        }
      }
  
      /**
       * Additional public API method to refresh the connection if still open (close, re-open).
       * For example, if the app suspects bad data / missed heart beats, it can try to refresh.
       */
      this.refresh = function() {
        if (ws) {
          ws.close()
        }
      }
    }
  
    /**
     * An event listener to be called when the WebSocket connection's readyState changes to OPEN;
     * this indicates that the connection is ready to send and receive data.
     */
    ReconnectingWebSocket.prototype.onopen = function(event) {}
    /** An event listener to be called when the WebSocket connection's readyState changes to CLOSED. */
    ReconnectingWebSocket.prototype.onclose = function(event) {}
    /** An event listener to be called when a connection begins being attempted. */
    ReconnectingWebSocket.prototype.onconnecting = function(event) {}
    /** An event listener to be called when a message is received from the server. */
    ReconnectingWebSocket.prototype.onmessage = function(event) {}
    /** An event listener to be called when an error occurs. */
    ReconnectingWebSocket.prototype.onerror = function(event) {}
  
    /**
     * Whether all instances of ReconnectingWebSocket should log debug messages.
     * Setting this to true is the equivalent of setting all instances of ReconnectingWebSocket.debug to true.
     */
    ReconnectingWebSocket.debugAll = false
  
    ReconnectingWebSocket.CONNECTING = WebSocket.CONNECTING
    ReconnectingWebSocket.OPEN = WebSocket.OPEN
    ReconnectingWebSocket.CLOSING = WebSocket.CLOSING
    ReconnectingWebSocket.CLOSED = WebSocket.CLOSED
  
    return ReconnectingWebSocket
  })
  