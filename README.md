# OLPart: Online Learning based Resource Partitioning for Colocating Multiple Latency-Critical Jobs on Commodity Computers
OLPart is tested on a CentOS 7.8 Server with Linux 4.1.0 using Python3.7. Please install the following library depencies for running OLPart.

# Dependencies
```
pip3 install numpy   
apt-get install intel-cmt-cat
```

OLPart uses the Linux _perf_ to collect runtime status of each job for guiding the quick and smart exploration. 
Please ensure that Intel CAT, MBA, and taskset tools are supported and active in your system.
Click this link to confirm: https://github.com/intel/intel-cmt-cat.

# The benchmark suites evaluated in Orchid

Tailbench: http://tailbench.csail.mit.edu

PARSEC 3.0: https://parsec.cs.princeton.edu/parsec3-doc.htm

# Run OLPart

## File Description
```
get_arm.py : generate resource configurations as arms.
get_config.py : some tool functions.
OLUCB.py: main algorithm of OLPart.
vote_bandit.py : main file used to make online resource partitioning decisions.
```

## run program:
    python vote_bandit.py
