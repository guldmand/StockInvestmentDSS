# k3s PoC Status

Verified on: 2026-05-11

## Cluster

- Platform: Turing Pi 2.4
- Nodes: 4 x RK1
- Kubernetes distribution: k3s
- k3s version: v1.34.4+k3s1
- Control plane: node1
- Workers: node2, node3, node4

## Node Status

All nodes were verified as Ready.

```text
NAME    STATUS   ROLES           INTERNAL-IP
node1   Ready    control-plane   10.0.0.213
node2   Ready    <none>          10.0.0.129
node3   Ready    <none>          10.0.0.179
node4   Ready    <none>          10.0.0.194
```

## System Pods

Core k3s services were running in the `kube-system` namespace.

Verified components included:

- coredns
- local-path-provisioner
- metrics-server
- traefik
- svclb-traefik

## PoC Namespace

The intended PoC namespace is:

```text
stockinvestmentdss
```

This namespace is used as the future deployment target for the StockInvestmentDSS system track.

## Minimal Workload Test

A temporary nginx workload was used to verify that the cluster can run a simple deployment.

Test scope:

- create a temporary deployment
- expose it through NodePort
- verify pod scheduling
- verify service creation
- test NodePort access where relevant
- remove the temporary workload afterwards

Example commands:

```bash
sudo k3s kubectl create namespace stockinvestmentdss

sudo k3s kubectl create deployment hello-k3s \
  --image=nginx \
  --replicas=4 \
  -n stockinvestmentdss

sudo k3s kubectl expose deployment hello-k3s \
  --port=80 \
  --type=NodePort \
  -n stockinvestmentdss

sudo k3s kubectl rollout status deployment/hello-k3s \
  -n stockinvestmentdss \
  --timeout=120s

sudo k3s kubectl get pods -n stockinvestmentdss -o wide
sudo k3s kubectl get svc -n stockinvestmentdss
```

Cleanup:

```bash
sudo k3s kubectl delete service hello-k3s -n stockinvestmentdss
sudo k3s kubectl delete deployment hello-k3s -n stockinvestmentdss
```

The `stockinvestmentdss` namespace should remain, since it is the future PoC namespace.

## NAS / DuckDB Storage Context

Persistent storage is provided by guldNAS.

Canonical NAS root:

```text
/mnt/nas/stockinvestmentdss
```

Canonical DuckDB database file path:

```text
/mnt/nas/stockinvestmentdss/duckdb/market_research.duckdb
```

DuckDB is embedded/in-process. It is not deployed as a standalone database server.

The database file is stored on guldNAS, while DuckDB query execution happens inside the process that opens the file, such as:

- local development scripts
- research notebooks
- backend containers
- ingestion workers
- feature workers
- decision workers
- training jobs
- Turing Pi / k3s workloads
- GPU box or cloud jobs

## Minimal Deployment Target

For the V1.0 PoC, the first Kubernetes deployment target is:

```text
namespace: stockinvestmentdss
cluster:   Turing Pi k3s
storage:   guldNAS-mounted persistent storage
```

The first real application deployment should be minimal:

```text
backend API
+ local/runtime DuckDB path or mounted NAS path
+ basic health endpoint
```

Frontend, ingress, monitoring, model training jobs and production deployment should come later.

## Scope

This task verifies the k3s platform foundation only.

It does not deploy the full StockInvestmentDSS application.

It confirms that the Turing Pi k3s cluster is available as a future staging/test/demo runtime platform for the system track.
