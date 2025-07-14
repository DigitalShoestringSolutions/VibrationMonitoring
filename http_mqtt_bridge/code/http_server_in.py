import json
import traceback
import cherrypy
import logging
import zmq
import threading
import multiprocessing
logger = logging.getLogger("main.http_server_in")
context = zmq.Context()

zmq_lock = threading.Lock()

class HTTPInBuildingBlock(multiprocessing.Process):
    def __init__(self, config, zmq_conf):
        super().__init__()

        self.config = config
        self.constants = config['constants']

        # declarations
        self.zmq_conf = zmq_conf

    def run(self):
        logger.info("started")
        cherrypy.config.update({
            'server.socket_host': '0.0.0.0',
            'server.socket_port': 8080})
        try:
            cherrypy.quickstart(PostHandler(self.zmq_conf), config={'global': {
                'engine.autoreload.on': False,
                # 'server.thread_pool':1
            }})
        except Exception:
            logger.error(traceback.format_exc())
        
        logger.info("Done")

def get_socket(zmq_conf):
    socket = context.socket(zmq_conf['type'])
    if zmq_conf["bind"]:
        socket.bind(zmq_conf["address"])
    else:
        socket.connect(zmq_conf["address"])
    return socket



def dispatch(config,output):
    socket = get_socket(config)
    logger.info(f"dispatch to { output['path']} of {output['payload']}")
    socket.send_json({'path': output.get('path', ""), 'payload': output['payload']})


class PostHandler(object):
    def __init__(self,zmq_conf):
        self.zmq_conf = zmq_conf

    @cherrypy.expose
    @cherrypy.tools.json_in()
    def default(self, *args):
        if cherrypy.request.method == "POST":
            topic = '/'.join(args)
            payload = cherrypy.request.json
            logger.debug(f"{topic}:>:{payload}")
            dispatch(self.zmq_conf,{'path':topic,'payload':payload})
            return
        
