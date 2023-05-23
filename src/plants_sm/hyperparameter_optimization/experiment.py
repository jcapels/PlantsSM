from abc import abstractmethod
from typing import Any

import optuna

from plants_sm.data_structures.dataset import Dataset


class Experiment:

    def __init__(self, train_dataset: Dataset, validation_dataset: Dataset = None, **kwargs):
        """
        Initialize an experiment.

        Parameters
        ----------
        kwargs: dict
            Keyword arguments to pass to optuna.create_study.
            storage:
                Database URL. If this argument is set to None, in-memory storage is used, and the
                :class:`~optuna.study.Study` will not be persistent.

                .. note::
                    When a database URL is passed, Optuna internally uses `SQLAlchemy`_ to handle
                    the database. Please refer to `SQLAlchemy's document`_ for further details.
                    If you want to specify non-default options to `SQLAlchemy Engine`_, you can
                    instantiate :class:`~optuna.storages.RDBStorage` with your desired options and
                    pass it to the ``storage`` argument instead of a URL.

                 .. _SQLAlchemy: https://www.sqlalchemy.org/
                 .. _SQLAlchemy's document:
                     https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
                 .. _SQLAlchemy Engine: https://docs.sqlalchemy.org/en/latest/core/engines.html
            sampler:
                A sampler object that implements background algorithm for value suggestion.
                If :obj:`None` is specified, :class:`~optuna.samplers.TPESampler` is used during
                single-objective optimization and :class:`~optuna.samplers.NSGAIISampler` during
                multi-objective optimization. See also :class:`~optuna.samplers`.
            pruner:
                A pruner object that decides early stopping of unpromising trials. If :obj:`None`
                is specified, :class:`~optuna.pruners.MedianPruner` is used as the default. See
                also :class:`~optuna.pruners`.
            study_name:
                Study's name. If this argument is set to None, a unique name is generated
                automatically.
            direction:
                Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for
                maximization. You can also pass the corresponding :class:`~optuna.study.StudyDirection`
                object. ``direction`` and ``directions`` must not be specified at the same time.

                .. note::
                    If none of `direction` and `directions` are specified, the direction of the study
                    is set to "minimize".
            load_if_exists:
                Flag to control the behavior to handle a conflict of study names.
                In the case where a study named ``study_name`` already exists in the ``storage``,
                a :class:`~optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
                set to :obj:`False`.
                Otherwise, the creation of the study is skipped, and the existing one is returned.
            directions:
                A sequence of directions during multi-objective optimization.
                ``direction`` and ``directions`` must not be specified at the same time.
        """
        self.best_parameters = None
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.study = optuna.create_study(**kwargs)

    @property
    @abstractmethod
    def best_experiment(self) -> Any:
        """
        Method to be implemented by all experiments to return the best experiment. It can be a Pipeline, a Model,
        a Transformer, etc.

        Returns
        -------
        Any
            The best experiment. It can be a Pipeline, a Model, Transformer, etc.
        """

    @abstractmethod
    def objective(self, trial: optuna.trial.Trial) -> float:
        """
        Method to be implemented by all experiments to define the objective function.

        Parameters
        ----------
        trial: optuna.trial.Trial
            A trial object that contains the current suggested hyperparameters.

        Returns
        -------
        float
            The value of the objective function.
        """

    def run(self, **kwargs):
        """
        Run the experiment.

        Parameters
        ----------
        kwargs: dict
            Keyword arguments to pass to optuna.Study.optimize.
            n_trials:
                The number of trials for each process. :obj:`None` represents no limit in terms of
                the number of trials. The study continues to create trials until the number of
                trials reaches ``n_trials``, ``timeout`` period elapses,
                :func:`~optuna.study.Study.stop` is called, or a termination signal such as
                SIGTERM or Ctrl+C is received.

                .. seealso::
                    :class:`optuna.study.MaxTrialsCallback` can ensure how many times trials
                    will be performed across all processes.
            timeout:
                Stop study after the given number of second(s). :obj:`None` represents no limit in
                terms of elapsed time. The study continues to create trials until the number of
                trials reaches ``n_trials``, ``timeout`` period elapses,
                :func:`~optuna.study.Study.stop` is called or, a termination signal such as
                SIGTERM or Ctrl+C is received.
            n_jobs:
                The number of parallel jobs. If this argument is set to ``-1``, the number is
                set to CPU count.

                .. note::
                    ``n_jobs`` allows parallelization using :obj:`threading` and may suffer from
                    `Python's GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`_.
                    It is recommended to use :ref:`process-based parallelization<distributed>`
                    if ``func`` is CPU bound.

            catch:
                A study continues to run even when a trial raises one of the exceptions specified
                in this argument. Default is an empty tuple, i.e. the study will stop for any
                exception except for :class:`~optuna.exceptions.TrialPruned`.
            callbacks:
                List of callback functions that are invoked at the end of each trial. Each function
                must accept two parameters with the following types in this order:
                :class:`~optuna.study.Study` and :class:`~optuna.trial.FrozenTrial`.

                .. seealso::

                    See the tutorial of :ref:`optuna_callback` for how to use and implement
                    callback functions.

            gc_after_trial:
                Flag to determine whether to automatically run garbage collection after each trial.
                Set to :obj:`True` to run the garbage collection, :obj:`False` otherwise.
                When it runs, it runs a full collection by internally calling :func:`gc.collect`.
                If you see an increase in memory consumption over several trials, try setting this
                flag to :obj:`True`.

                .. seealso::

                    :ref:`out-of-memory-gc-collect`

            show_progress_bar:
                Flag to show progress bars or not. To disable progress bar, set this :obj:`False`.
                Currently, progress bar is experimental feature and disabled
                when ``n_trials`` is :obj:`None`, ``timeout`` not is :obj:`None`, and
                ``n_jobs`` :math:`\\ne 1`.

        """
        self.study.optimize(self.objective, **kwargs)
        self.best_parameters = self.study.best_params
