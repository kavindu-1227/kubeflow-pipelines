#!/bin/bash
CLUSTER_NAME="cluster-6"
ZONE="us-west1-a"
MACHINE_TYPE="e2-standard-2" # A machine with 2 CPUs and 8GB memory.
SCOPES="cloud-platform"

gcloud container clusters create $CLUSTER_NAME \
     --zone $ZONE \
     --machine-type $MACHINE_TYPE \
     --scopes $SCOPES
     
echo Kubernetes cluster created in GCP successfully
sleep 3m
     
export PIPELINE_VERSION=1.8.5
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
sleep 3m

kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
sleep 3m

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"

echo Kubeflow Pipelines installed on Kubernetes cluster successfully

sleep 5m

URL=$(kubectl describe configmap inverse-proxy-config -n kubeflow | grep googleusercontent.com)

python3 kubeflow-pipeline-all.py --param1 $URL
echo kubeflow-pipeline.py started execution ::: cluster will be deleted in 2hrs

sleep 2h

gcloud container clusters delete $CLUSTER_NAME --zone=$ZONE --quiet

echo Kubernetes cluster deleted successfully