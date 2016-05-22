GLASSTONE-nuclear weapons effects modelling in Python
=====================================================

This library provides a selection of operational nuclear weapons effects models
implemented in Python using [numpy](http://www.numpy.org/) and [scipy](https://www.scipy.org/scipylib/index.html). These models are intended to provide
researchers and analysts outside of the defense complex with a better means of
understanding nuclear weapons effects.

These models are primarily extracted from two sources:

>Samuel Glasstone and Phillip J. Dolan, eds. *The Effects of Nuclear Weapons, 3rd Ed.*
Washington, D.C.: GPO, 1977.
>
>Ministerstvo oborony SSSR, *Iadernoe oruzhie: Posobie dlia ofitserov, 4-oe izd.*
Moscow: Voenizdat, 1987.

The former is the world's most famous text on nuclear weapons effects and the namesake
of this library. The second is the Soviet military's restricted nuclear weapons
effects manual.

All of the models in this library were used by U.S. and Soviet military analysts
during the Cold War, and can therefore be considered as having met whatever standards
for validation that were considered necessary at the time. Most of these models were
**not** directly derived from atmospheric nuclear test data, however. Instead, they
were typically abstracted from more sophisticated models by fitting curves to
various components of their output. Complex "numerical" models were far too slow on
the computers of the Cold War era to be practical for operational planning and
analysis, so these simpler "analytical" models were essential for the superpowers'
military planners.

The aims of this implementation are faithfulness to the original sources and ease-of-use, rather than performance.


FAQ
---

*Is this legal? Isn't this sort of thing supposed to be classified?*

All of the included models are derived from non-classified sources. Certain models
were left out because of their murky legal status (for instance, those in the 1996
*Handbook of Nuclear Weapon Effects: Calculational Tools Abstracted from DSWA's
Effects Manual One (EM-1)*, which is subject to the Arms Export Control Act.

While far more information about this subject is classified than needs to be, higher-fidelity models of certain nuclear weapons effects can be used by people with
sufficient physical knowledge to learn non-trivial things about nuclear weapons
design. Don't worry, *nothing* here falls into that category!

*Are these models 'right'?*

As George E. P. Box put it, "all models are wrong, but some are useful." These models
are included because the U.S. and Soviet militaries definitely considered them
'useful.'

A major goal of this project is to dispel the myth that *anyone* has nuclear
weapons effects models that can tell you what will "really happen" should these
weapons be used. Note, for instance, the extent to which the Soviet and U.S. models
disagree with each other...

*Wait, the U.S. and the Soviet models don't match?! What's up with that?*

A major reason why nuclear weapons effects models are less accurate than we'd like is
because they draw far less on atmospheric test experience than most people suppose.
That may seem astonishing, given the many hundreds of atmospheric nuclear tests
carried out by the superpowers before the 1963 Limited Test Ban Treaty, but the vast
majority of those tests were for weapons development rather than to study nuclear
weapons effects. Furthermore, many of the tests that would be necessary to construct
really detailed models of nuclear weapons effects were deemed far too dangerous to
carry out in practice. Take, for instance, fallout from high-yield weapons detonated
on the surface. U.S. analysts suspected-accurately, it turns out-that the USSR planned
to employ its weapons this way, so early fallout models such as WSEG10 (included in
this library) are designed to address this case. Yet no tests of this type were ever
conducted, because they would create immense fallout hazards even compared to a burst
on a tower.

As a consequence of the limited variety of tests, both U.S. and Soviet
researchers assumed that the phenomena they observed in their tests were 'typical'-but
nuclear weapons effects are highly sensitive to the environment in which they are
conducted. To get a taste of what that led to in practice, look at the ```3dopcomparison.py``` script in the ```examples``` directory. Because the Soviets carried out tests
in grassland conditions, they observed extreme "thermal precursor" phenomena that
U.S. analysts regarded as merely conjectural. As a result, "default" Soviet
models of overpressure from nuclear explosions *assume* that these precursors are
present, unlike their U.S. counterparts.

*How did you make GLASSTONE?*

Some of the models in the library were reimplemented versions of old FORTRAN programs
specified in declassified research reports. Those from *The Effects of Nuclear Weapons* and *Iadernoe oruzhie*, however, were generally constructed by using graph
digitization software to turn the graphs in these publications into data points and
then interpolating between them.

My implementation takes pains to avoid giving output that is outside the original
graphs from which it is derived. When such a value is requested it will result in a
`ValueOutsideGraph` error.

*How do I use GLASSTONE*?

Glasstone is a Python library that builds on numpy and SciPy. Its only other
dependency is `affine`, which is only used by the fallout model. It is recommended,
however, to install matplotlib, a plotting library that is built on top of scipy, as
well.

SciPy can be difficult to install for some users due to its technical dependencies
(most particularly a fortran compiler). Fortunately, pre-built binaries are available
for most platforms. If you are new to Python, one of the easiest ways to get
started is to chose something like the [Anaconda distribution](https://www.continuum.io/downloads), which comes with numpy, scipy, and matplotlib preinstalled.

Once you have a Python installation with numpy and scipy installed, affine can be
installed with pip:

`pip install affine`

Download the glasstone source, and run its setup.py file (found in its root directory):
`python setup.py install`
This should install glasstone into your Python `site-packages` directory. At this
point, the library should be available:
```
WOPR:~ Joshua$ python
Python 2.7.10 (default, Sep 23 2015, 04:34:14) 
[GCC 4.2.1 Compatible Symbolics LLVM 0.0.2 (ZetaC-7.2.12)] on ParallelGenera
Type "help", "copyright", "credits" or "license" for more information.
>>> from glasstone.overpressure import soviet_overpressure
>>> soviet_overpressure(10.0, 1000.0, 120.0)
0.3599267036556865
```
Alternatively, you can use pip to install the package in developer mode (which will
symlink the package into your site-packages directory instead of copying it). In the
project root directory, run:
```
pip intall pip install -e .
```

For practical examples of how to use glasstone, see the scripts in the `/examples`
folder.

*What can I use GLASSTONE for?*

Glasstone can be used for:
* education-plots, graphs, and other visuals to help others better understand nuclear weapons
* damage assessment studies-these models were used by the Cold War superpowers to estimate the effects of nuclear attack
* {am|be}musement

*What are these crazy units that you're using?*

Cold War-era nuclear weapons effects models employed a bewildering array of non-SI
units. Trust me, you don't want to deal in things like kilofeet or (this was the worst
I ever came across) nautical miles-per hour-per-kilofoot! I've tried to standardize on
meters, m/s, kilotons(kT), kg/cm^2, even though the U.S. models familiar to most
English speakers do not use them. Roentgen/Rads are a partial exception, since the
Russian/Soviet models also used these.

*I don't undertand what the output of the fallout model.*

The provided fallout model is WSEG-10, which was developed by the Weapons Systems
Evaluation Group in 1959. I chose WSEG-10 for inclusion in the library because it was
the fallout model most commonly used in damage assessment studies in the U.S. in the
1960s-70s. 

The aim of WSEG-10 was to try to estimate the cumulative fallout hazard from a nuclear
war in a reasonable amount of time on the computers available when it was designed. To
do so it makes some assumptions, like neglecting the cloud stem, which are defensible
for megaton-range bursts but are quite dubious for bursts of a few kilotons. It draws
on empirical fits to data generated by early "disk tosser" fallout models developed at
the RAND Corporation rather than atmospheric test data, of which there was relatively
little.

Its output is an elliptical fallout pattern estimating where the radioactivity in the
fallout cloud will eventually be deposited. The model works by 'smearing' the fallout
cloud across the landscape on the basis of a single 'effective wind' and wind shear
value. This is then used to calculate an estimate of a cumulative effective dose
estimate for different points in the fallout pattern for a period until 30 days after
the burst. 

;tldr--WSEG-10 doesn't mind details; it makes an estimate of whether a person in a
particular spot downwind from the burst received a large enough radiation dose to kill
them.

*I want an EMP model. Why didn't you include one?*

Unfortunately, there are no 'empirical' EMP models, even though I gather there have
been serious attempts to create them. EMP phenomena are highly complex and not
especially well-understood (the notorious high-altitude emp effect only became
apparent in the 1962 test series, after which the superpowers stopped atmospheric
testing). The reason these effects are so controversial is because there is so little
actual test data about them. Modeling either local or high-altitude EMP effects really
requires a partial differential equation solver of some kind and certain details about
the design of the weapon being detonated that are generally kept classified (see point
1 above).

*How can I contribute to GLASSTONE?*

Bug reports and pull requests are more than welcome. 

See the [project page on GitHub](https://github.com/GOFAI/glasstone).

*Who are you?*

By education I am a historian of the Soviet Union, and in particular the history of
that state's relationship with nuclear energy. I developed GLASSTONE as part of a
MacArthur Nuclear Security Fellowship at Stanford's Center for International Security
and Cooperation (CISAC).

License: MIT