# Contributing to Ignnition

Thank you for your interest in contributing to Ignnition! If you wish to contribute, please abide by our branching 
system.

## Managing branches in Ignnition

There are three main branches in Ignnition:

- **main**: The main branch of Ignnition. This branch contains the latest stable release.
- **ignnition-nightly**: The nightly branch of Ignnition. This branch contains the small incremental updates that are 
released in a constant release cycle.
- **development**: The development branch of Ignnition. This branch contains future features that may drastically change
how Ignnition is used. As such, this branch is expected to be unstable and should not be used for any other use than alpha testing.
  
There are three different kinds of development branches:

- **hf-(name)**: This branch contains hotfixes (i.e. fixes deemed urgent enough to be included directly into the main 
branch without having to wait to the end of the development cycle). These branches should start from the **main** 
branch and its PR should be directed at the **main** branch. The hotfix version should be increased whenever one of these 
branches is created.
  
- **bf-(name)**: This branch contains bugfixes (i.e. fixes not deemed urgent enough to be included directly into the 
main branch and that can wait until the development cycle ends). These branches should start from the **ignnition-nightly**
branch and its PR should be directed at the **ignnition-nightly** branch.

- **ft-(name)**: This branch contains new features. These branches should start from the **ignnition-nightly** branch
and its PR should be directed at the **ignnition-nightly** branch.
  
- **dev-(name)**: This branch contains new features. These features are expected to contain deeper changes than those in
the **ft-(name)** branches. These branches should start from the **development** branch and its PR should be directed to
the **development** branch.

## Versioning

The *_version.py* file inside the *ignnition* folder contains the version of the package. The version follows a format 
"x.y.z", in which *x* is the **major** version, *y* is the **minor** version and *z* is the **fix** version.

The **fix** version is increased with every **hf-(name)** branch. The expected workflow should be the following: create 
a new hotfix branch, increase the fix version and create a PR to the **main** branch. After that, the performed changes 
should be ported to the nightly branch with a cherry-pick to ensure that the hotfix is also present there.

The **minor** version is increased at the end of every development cycle. The expected workflow should be the following:
at the beginning of every development cycle, increase the minor version of the **ignition-nightly** branch. At the end 
of the development cycle, create a PR to the **main** branch to bring all the changes made during the cycle to the
next release. 
Increasing this version resets the **fix** version.

The **major** version is increased with every new **development** branch. The expected workflow should be the following:
whenever a new **development** branch is created, increase the major version. Whenever it is deemed necessary, create a 
PR to the **main** branch to bring all the changes to the next release.
Increasing this version resets the **minor** and **fix** versions.
