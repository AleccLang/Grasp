class ActionController:

    """GraspEngine class contains all the methods necessary for the proccessing and evaluation of images.
    
    ...
    
    Methods
    -------
    recordNewAction(action_type, action_indexes, action_items):
        Records a new action made by the user and stores data needed to undo or redo it.

    undoLastAction():
        Records that the last action as been undone and returns data needed to undo the action.

    redoLastUndo():
        Records that the last undone action has been redone and returns data needed to redo the action.
    """

    def __init__(self):
        self.action_history = []
        self.thing_action_history = [[] for i in range(999)]
        self.thing_action_indexes = [-1 for i in range(999)]
        self.thing_action_types = [0 for i in range(999)]
        self.image_action_history = [[] for i in range(999)]
        self.image_action_indexes = [-1 for i in range(999)]
        self.image_indexes = []
        self.thing_indexes = []
        self.current_action_index = -1

    def recordNewAction(
        self,
        action_type,
        action_indexes,
        action_items,
        ):
        """Records a new action made by the user and stores data needed to undo or redo it.
        
        Parameters
        ----------
        action_type : Identifies the type of action being recorded.
        action_indexes : Contains the indexes of all items included in the action.
        action_items : Contains the current state of the items included in the action.
        """

        self.current_action_index += 1
        item_indexes = []
        for (i, index) in enumerate(action_indexes):
            (item_type, item_index) = index
            item_indexes.append(item_index)
            if item_type == 'Hand':
                self.image_action_history[item_index] = \
                    (self.image_action_history[item_index])[:self.image_action_indexes[item_index]
                    + 1]
                self.image_action_history[item_index].append((action_type,
                        action_items[i]))
                self.image_indexes.append(item_index)
                self.image_action_indexes[item_index] += 1
            else:
                self.thing_action_history[item_index] = \
                    (self.thing_action_history[item_index])[:self.thing_action_indexes[item_index]
                    + 1]
                self.thing_action_history[item_index].append((action_type,
                        action_items[i]))
                self.thing_indexes.append(item_index)
                self.thing_action_types[item_index] = item_type
                self.thing_action_indexes[item_index] += 1
        self.action_history = \
            self.action_history[:self.current_action_index]
        self.action_history.append(item_indexes)

    def undoLastAction(self):
        """Records that the last action as been undone and returns data needed to undo the action.
        
        Returns
        -------
        previous_action_type: Identifies the type of action which was previously made.
        current_action_type: Identifies the type of the current action.
        previous_return_indexes: Contains the indexes of all items included in the previous action.
        current_return_indexes: Contains the indexes of all items included in the current action.
        previous_items: Contains the state of all items in the previous action.
        current_items: Contains the state of all items in the current action.
        """

        current_indexes = self.action_history[self.current_action_index]
        previous_indexes = \
            self.action_history[self.current_action_index - 1]
        previous_action_type = ''
        current_action_type = ''
        previous_items = []
        current_items = []
        previous_return_indexes = []
        current_return_indexes = []
        for index in previous_indexes:
            if index in self.image_indexes:
                (typ, it) = \
                    self.image_action_history[index][self.image_action_indexes[index]
                        - 1]
                item_type = 'Hand'
                previous_return_indexes.append((item_type, index))
            elif index in self.thing_indexes:
                (typ, it) = \
                    self.thing_action_history[index][self.thing_action_indexes[index]
                        - 1]
                item_type = self.thing_action_types[index]
                previous_return_indexes.append((item_type, index))
            previous_action_type = typ + ''
            previous_items.append(it)
        for index in current_indexes:
            if index in self.image_indexes:
                (typ, it) = \
                    self.image_action_history[index][self.image_action_indexes[index]]
                item_type = 'Hand'
                current_return_indexes.append((item_type, index))
                self.image_action_indexes[index] = \
                    self.image_action_indexes[index] - 1
            elif index in self.thing_indexes:
                (typ, it) = \
                    self.thing_action_history[index][self.thing_action_indexes[index]]
                item_type = self.thing_action_types[index]
                current_return_indexes.append((item_type, index))
                self.thing_action_indexes[index] = \
                    self.thing_action_indexes[index] - 1
            current_action_type = typ + ''
            current_items.append(it)
        self.current_action_index -= 1
        return (
            previous_action_type,
            current_action_type,
            previous_return_indexes,
            current_return_indexes,
            previous_items,
            current_items,
            )

    def redoLastUndo(self):
        """Records that the last undone action has been redone and returns data needed to redo the action.
        
        Returns
        -------
        undone_action_type: Identifies the type of action which was undone.
        current_action_type: Identifies the type of the current action.
        undone_return_indexes: Contains the indexes of all items included in the undone action.
        current_return_indexes: Contains the indexes of all items included in the current action.
        undone_items: Contains the state of all items in the undone action.
        current_items: Contains the state of all items in the current action.
        """

        self.current_action_index += 1
        undone_indexes = self.action_history[self.current_action_index]
        current_indexes = self.action_history[self.current_action_index
                - 1]
        undone_action_type = ''
        current_action_type = ''
        undone_items = []
        current_items = []
        undone_return_indexes = []
        current_return_indexes = []
        for index in current_indexes:
            if index in self.image_indexes:
                (typ, it) = \
                    self.image_action_history[index][self.image_action_indexes[index]]
                item_type = 'Hand'
                current_return_indexes.append((item_type, index))
            elif index in self.thing_indexes:
                (typ, it) = \
                    self.thing_action_history[index][self.thing_action_indexes[index]]
                item_type = self.thing_action_types[index]
                current_return_indexes.append((item_type, index))
            current_action_type = typ + ''
            current_items.append(it)
        for index in undone_indexes:
            if index in self.image_indexes:
                (typ, it) = \
                    self.image_action_history[index][self.image_action_indexes[index]
                        + 1]
                item_type = 'Hand'
                undone_return_indexes.append((item_type, index))
                self.image_action_indexes[index] = \
                    self.image_action_indexes[index] + 1
            elif index in self.thing_indexes:
                (typ, it) = \
                    self.thing_action_history[index][self.thing_action_indexes[index]
                        + 1]
                item_type = self.thing_action_types[index]
                undone_return_indexes.append((item_type, index))
                self.thing_action_indexes[index] = \
                    self.thing_action_indexes[index] + 1
            undone_action_type = typ + ''
            undone_items.append(it)
        return (
            undone_action_type,
            current_action_type,
            undone_return_indexes,
            current_return_indexes,
            undone_items,
            current_items,
            )
