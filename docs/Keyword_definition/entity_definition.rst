Step 1: Entity definition
^^^^^^^^^^^^^^^^^^^^^^^^^

In order to create the entities, we must define a list of "entities". For this, we must define an object "Entity".
We shall now describe the different keywords that the user must / can define to model the new entity, these being:


.. contents::
    :local:
    :depth: 1


----

Parameter: name
~~~~~~~~~~~~~~~

**Description:** Name that we assign to the new entity. This name is important as we will use it from now on to reference the nodes that belong to this entity.

**Accepted values:** String of the choice of the user.

E.g., below we show how we would define an entity of name *entity1*.

.. code-block:: yaml

   name: entity1

----

Parameter: state_dim
~~~~~~~~~~~~~~~~~~~~

**Description:** Dimension of the hidden states of the nodes of this entity.

**Accepted values:** Natural number

.. code-block:: yaml

   state_dim: 32

----

Parameter: initial_state
~~~~~~~~~~~~~~~~~~~~~~~~

**Description:** Array of Operation object defining incrementally the initial_state.

**Accepted values:** Array of :ref:`Operation objects <operation-object>`.
