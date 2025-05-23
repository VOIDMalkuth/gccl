/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include "glog/logging.h"
#include "utils.h"

namespace gccl {

#define MAX_IF_NAME_SIZE 32
#define MAX_IFS 16
#define SLEEP_INT 1000   // sleep interval in usec
#define RETRY_TIMES 2e4  // retry times before reporting a timeout (20 sec)

/* Common socket address storage structure for IPv4/IPv6 */
union socketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

/* Format a string representation of a (struct sockaddr *) socket address using
 * getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
static inline const char* socketToString(struct sockaddr* saddr, char* buf) {
  if (buf == NULL || saddr == NULL) return NULL;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) {
    buf[0] = '\0';
    return buf;
  }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  (void)getnameinfo(saddr, sizeof(union socketAddress), host, NI_MAXHOST,
                    service, NI_MAXSERV, NI_NUMERICHOST | NI_NUMERICSERV);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static inline int envSocketFamily(void) {
  int family = -1;  // Family selection is not forced, will use first one found
  char* env = getenv("NCCL_SOCKET_FAMILY");
  if (env == NULL) return family;

  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET;  // IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6;  // IPv6
  return family;
}

static int findInterfaces(const char* prefixList, char* names,
                          union socketAddress* addrs, int sock_family,
                          int maxIfNameSize, int maxIfs) {
  char line[1024];
  struct netIf userIfs[maxIfs];
  bool searchNot = prefixList && prefixList[0] == '^';
  int nUserIfs = parseStringList(prefixList, userIfs, maxIfs);

  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && found < maxIfs;
       interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) continue;

    DLOG(INFO) << "Found interface " << interface->ifa_name << ":"
              << socketToString(interface->ifa_addr, line);

    /* Allow the caller to force the socket family type */
    if (sock_family != -1 && family != sock_family) continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
    }

    // check against user specified interfaces
    if (!(matchIfList(interface->ifa_name, -1, userIfs, nUserIfs) ^
          searchNot)) {
      continue;
    }

    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    bool duplicate = false;
    for (int i = 0; i < found; i++) {
      if (strcmp(interface->ifa_name, names + i * maxIfNameSize) == 0) {
        duplicate = true;
        break;
      }
    }

    if (!duplicate) {
      // Store the interface name
      strncpy(names + found * maxIfNameSize, interface->ifa_name,
              maxIfNameSize);
      // Store the IP address
      int salen =
          (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
      memcpy(addrs + found, interface->ifa_addr, salen);
      DLOG(INFO) << "NET : Using interface " << interface->ifa_name << ":"
                << socketToString(interface->ifa_addr, line);
      found++;
    }
  }

  freeifaddrs(interfaces);
  return found;
}

static bool matchSubnet(struct ifaddrs local_if, union socketAddress remote) {
  /* Check family first */
  int family = local_if.ifa_addr->sa_family;
  if (family != remote.sa.sa_family) {
    return false;
  }

  if (family == AF_INET) {
    struct sockaddr_in* local_addr = (struct sockaddr_in*)(local_if.ifa_addr);
    struct sockaddr_in* mask = (struct sockaddr_in*)(local_if.ifa_netmask);
    struct sockaddr_in& remote_addr = remote.sin;
    struct in_addr local_subnet, remote_subnet;
    local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;
    return (local_subnet.s_addr ^ remote_subnet.s_addr) ? false : true;
  } else if (family == AF_INET6) {
    struct sockaddr_in6* local_addr = (struct sockaddr_in6*)(local_if.ifa_addr);
    struct sockaddr_in6* mask = (struct sockaddr_in6*)(local_if.ifa_netmask);
    struct sockaddr_in6& remote_addr = remote.sin6;
    struct in6_addr& local_in6 = local_addr->sin6_addr;
    struct in6_addr& mask_in6 = mask->sin6_addr;
    struct in6_addr& remote_in6 = remote_addr.sin6_addr;
    bool same = true;
    int len = 16;                    // IPv6 address is 16 unsigned char
    for (int c = 0; c < len; c++) {  // Network byte order is big-endian
      char c1 = local_in6.s6_addr[c] & mask_in6.s6_addr[c];
      char c2 = remote_in6.s6_addr[c] & mask_in6.s6_addr[c];
      if (c1 ^ c2) {
        same = false;
        break;
      }
    }
    // At last, we need to compare scope id
    // Two Link-type addresses can have the same subnet address even though they
    // are not in the same scope For Global type, this field is 0, so a
    // comparison wouldn't matter
    same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
    return same;
  } else {
    LOG(ERROR) << "Net : Unsupported address family type";
    return false;
  }
}

static int findInterfaceMatchSubnet(char* ifNames,
                                    union socketAddress* localAddrs,
                                    union socketAddress remoteAddr,
                                    int ifNameMaxSize, int maxIfs) {
  char line[1024], line_a[1024];
  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && !found;
       interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) continue;

    // check against user specified interfaces
    if (!matchSubnet(*interface, remoteAddr)) {
      continue;
    }

    // Store the local IP address
    int salen =
        (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
    memcpy(localAddrs + found, interface->ifa_addr, salen);

    // Store the interface name
    strncpy(ifNames + found * ifNameMaxSize, interface->ifa_name,
            ifNameMaxSize);

    LOG(INFO) << "NET : Found interface " << interface->ifa_name << ":"
              << socketToString(&(localAddrs[found].sa), line)
              << " in the same subnet as remote address "
              << socketToString(&(remoteAddr.sa), line_a);
    found++;
    if (found == maxIfs) break;
  }

  if (found == 0) {
    LOG(ERROR)
        << "Net : No interface found in the same subnet as remote address "
        << socketToString(&(remoteAddr.sa), line_a);
  }
  freeifaddrs(interfaces);
  return found;
}

static gcclResult_t GetSocketAddrFromString(union socketAddress* ua,
                                            const char* ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    LOG(ERROR) << "Net : string is null";
    return gcclInvalidArgument;
  }

  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      LOG(ERROR) << "Net : No valid <IPv4_or_hostname>:<port> pair found";
      return gcclInvalidArgument;
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ((rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      LOG(ERROR) << "Net : error encountered when getting address info : "
                 << gai_strerror(rv);
      return gcclInvalidArgument;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in& sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET;  // IPv4
      // inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port);  // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6& sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;      // IPv6
      sin6.sin6_port = htons(ni.port);  // port
      sin6.sin6_flowinfo = 0;           // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;           // should be global scope, set to 0
    } else {
      LOG(ERROR) << "Net : unsupported IP family";
      return gcclInvalidArgument;
    }

    freeaddrinfo(p);  // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%') j = i;
      if (ip_port_pair[i] == ']') break;
    }
    if (i == len) {
      LOG(ERROR) << "Net : No valid [IPv6]:port pair found";
      return gcclInvalidArgument;
    }
    bool global_scope =
        (j == -1
             ? true
             : false);  // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair + 1, global_scope ? i - 1 : j - 1);
    strncpy(port_str, ip_port_pair + i + 2, len - i - 1);
    int port = atoi(port_str);
    if (!global_scope)
      strncpy(if_name, ip_port_pair + j + 1,
              i - j - 1);  // If not global scope, we need the intf name

    struct sockaddr_in6& sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                     // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));  // IP address
    sin6.sin6_port = htons(port);                    // port
    sin6.sin6_flowinfo = 0;  // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id =
        global_scope
            ? 0
            : if_nametoindex(
                  if_name);  // 0 if global scope; intf index if link scope
  }
  return gcclSuccess;
}

static int findInterfaces(char* ifNames, union socketAddress* ifAddrs,
                          int ifNameMaxSize, int maxIfs) {
  int nIfs = 0;
  // Allow user to force the INET socket family selection
  int sock_family = envSocketFamily();
  // User specified interface
  char* env = getenv("NCCL_SOCKET_IFNAME");
  if (env && strlen(env) > 1) {
    // Specified by user : find or fail
    nIfs = findInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize,
                          maxIfs);
  } else {
    // Try to automatically pick the right one
    // Start with IB
    nIfs = findInterfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize,
                          maxIfs);
    // else see if we can get some hint from COMM ID
    if (nIfs == 0) {
      char* commId = getenv("NCCL_COMM_ID");
      if (commId && strlen(commId) > 1) {
        // Try to find interface that is in the same subnet as the IP in comm id
        union socketAddress idAddr;
        GetSocketAddrFromString(&idAddr, commId);
        nIfs = findInterfaceMatchSubnet(ifNames, ifAddrs, idAddr, ifNameMaxSize,
                                        maxIfs);
      }
    }
    // Then look for anything else (but not docker or lo)
    if (nIfs == 0)
      nIfs = findInterfaces("^docker,lo", ifNames, ifAddrs, sock_family,
                            ifNameMaxSize, maxIfs);
    // Finally look for docker, then lo.
    if (nIfs == 0)
      nIfs = findInterfaces("docker", ifNames, ifAddrs, sock_family,
                            ifNameMaxSize, maxIfs);
    if (nIfs == 0)
      nIfs = findInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize,
                            maxIfs);
  }
  return nIfs;
}

static gcclResult_t createListenSocket(int* fd,
                                       union socketAddress* localAddr) {
  /* IPv4/IPv6 support */
  int family = localAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);

  /* Create socket and bind it to a port */
  int sockfd = socket(family, SOCK_STREAM, 0);
  if (sockfd == -1) {
    LOG(ERROR) << "Net : Socket creation failed : " << strerror(errno);
    return gcclSystemError;
  }

  int opt = 1;
  SYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                      sizeof(opt)),
           "setsockopt");

  // localAddr port should be 0 (Any port)
  SYSCHECK(bind(sockfd, &localAddr->sa, salen), "bind");

  /* Get the assigned Port */
  socklen_t size = salen;
  SYSCHECK(getsockname(sockfd, &localAddr->sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[1024];
  TRACE(INIT | NET, "Listening on socket %s",
        socketToString(&localAddr->sa, line));
#endif

  /* Put the socket in listen mode */
  SYSCHECK(listen(sockfd, 128), "listen");
  *fd = sockfd;
  return gcclSuccess;
}

static gcclResult_t connectAddress(int* fd, union socketAddress* remoteAddr) {
  /* IPv4/IPv6 support */
  int family = remoteAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);

  /* Connect to a hostname / port */
  *fd = socket(family, SOCK_STREAM, 0);
  if (*fd == -1) {
    LOG(ERROR) << "Net : Socket creation failed : %s", strerror(errno);
    return gcclSystemError;
  }

  const int one = 1;
  SYSCHECK(setsockopt(*fd, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)),
           "setsockopt");

  /*  const int bufsize = 128*1024;
    SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_SNDBUF, (char*)&bufsize,
    sizeof(int)), "setsockopt");
    SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_RCVBUF, (char*)&bufsize,
    sizeof(int)), "setsockopt");*/

  char line[1024];
  socketToString(&remoteAddr->sa, line);
  DLOG(INFO) << "Connecting to socket " << line;

  SYSCHECKNTIMES(connect(*fd, &remoteAddr->sa, salen), "connect", RETRY_TIMES,
                 SLEEP_INT, ECONNREFUSED);
  return gcclSuccess;
}

// static gcclResult_t socketReceive(int fd, void* ptr, size_t size) {
//   char* data = (char*)ptr;
//   size_t offset = 0;
//   while (offset < size) {
//     long long recvsize;
//     SYSCHECKVAL(recv(fd, data, size - offset, 0), "recv", recvsize);
//     if (recvsize == 0) {
//       LOG(ERROR) << "Net : Connection closed by remote peer";
//       return gcclSystemError;
//     }
//     if (recvsize == -1) {
//       LOG(ERROR) << "Recv : got retcode " << errno << ", retrying";
//       continue;
//     }
//     data += recvsize;
//     offset += recvsize;
//   }
//   return gcclSuccess;
// }

// static gcclResult_t socketSend(int fd, void* ptr, size_t size) {
//   char* data = (char*)ptr;
//   size_t offset = 0;
//   while (offset < size) {
//     long long sendsize;
//     SYSCHECKVAL(write(fd, data, size - offset), "write", sendsize);
//     if (sendsize == -1) {
//       LOG(ERROR) << "Send : got retcode " << errno << ", retrying";
//       continue;
//     }
//     data += sendsize;
//     offset += sendsize;
//   }
//   return gcclSuccess;
// }

#define GCCL_SOCKET_SEND 0
#define GCCL_SOCKET_RECV 1
static gcclResult_t socketProgress(int op, int fd, void* ptr, int size, int* offset) {
  int bytes = 0;
  char* data = (char*)ptr;
  do {
    if (op == GCCL_SOCKET_RECV) bytes = recv(fd, data+(*offset), size-(*offset), MSG_DONTWAIT);
    if (op == GCCL_SOCKET_SEND) bytes = send(fd, data+(*offset), size-(*offset), MSG_DONTWAIT);
    if (op == GCCL_SOCKET_RECV && bytes == 0) {
      LOG(ERROR) << "Net : Connection closed by remote peer";
      return gcclSystemError;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        LOG(ERROR) << "Call to recv failed : " << strerror(errno);
        return gcclSystemError;
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;
  } while (bytes > 0 && (*offset) < size);
  return gcclSuccess;
}

static gcclResult_t socketWait(int op, int fd, void* ptr, int size, int* offset) {
  while (*offset < size)
    GCCLCHECK(socketProgress(op, fd, ptr, size, offset));
  return gcclSuccess;
}

static gcclResult_t socketSend(int fd, void* ptr, int size) {
  int offset = 0;
  GCCLCHECK(socketWait(GCCL_SOCKET_SEND, fd, ptr, size, &offset));
  return gcclSuccess;
}

static gcclResult_t socketReceive(int fd, void* ptr, int size) {
  int offset = 0;
  GCCLCHECK(socketWait(GCCL_SOCKET_RECV, fd, ptr, size, &offset));
  return gcclSuccess;
}

}  // namespace gccl
