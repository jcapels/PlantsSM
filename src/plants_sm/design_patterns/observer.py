from __future__ import annotations
from abc import ABC, abstractmethod


class Subject(ABC):
    """
    The Subject interface declares a set of methods for managing subscribers.
    """

    @abstractmethod
    def attach(self, observer: Observer) -> None:
        """
        Attach an observer to the subject.
        """

    @abstractmethod
    def detach(self, observer: Observer) -> None:
        """
        Detach an observer from the subject.
        """

    @abstractmethod
    def notify(self) -> None:
        """
        Notify all observers about an event.
        """


class ConcreteSubject(Subject):
    """
    The Subject owns some important state and notifies observers when the state
    changes.
    """

    _state: int = None
    """
    For the sake of simplicity, the Subject's state, essential to all
    subscribers, is stored in this variable.
    """

    _observer: Observer = None
    """
    List of subscribers. In real life, the list of subscribers can be stored
    more comprehensively (categorized by event type, etc.).
    """

    def attach(self, observer: Observer) -> None:
        self._observer = observer

    def detach(self, observer: Observer) -> None:
        self._observer = None

    """
    The subscription management methods.
    """

    def notify(self, **kwargs) -> None:
        """
        Trigger an update in each subscriber.
        """

        self._observer.update(self, **kwargs)

    def end(self):
        self._observer.end()


class Observer(ABC):
    """
    The Observer interface declares the update method, used by subjects.
    """

    @abstractmethod
    def update(self, subject: Subject, **kwargs) -> None:
        """
        Receive update from subject.
        """

    @abstractmethod
    def end(self):
        """
        End the observer.
        """
