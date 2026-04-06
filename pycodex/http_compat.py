try:
    from http.server import ThreadingHTTPServer
except ImportError:  # pragma: no cover - Python 3.6 path
    from http.server import HTTPServer
    from socketserver import ThreadingMixIn

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
