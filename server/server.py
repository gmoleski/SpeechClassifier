from twisted.internet import reactor, protocol
import common

class Server(protocol.Protocol):
    pass

class ServerFactory(protocol.Factory):
    protocol = Server

factory = ServerFactory()
reactor.listenTCP(common.PORT, factory)
reactor.run()