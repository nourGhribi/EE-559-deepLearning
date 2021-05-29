import math
import torch
from losses import LossMSE as MSE
from optimizers import SGD

def get_data(n=1000):
    """
    Returns (x, label) n data points with x sampled uniformly in [0, 1]^2,
    each with a label 1 if outside the disk centered at (0.5, 0.5) of radius 1/√2π, and 0 if inside.
    :param n: Number of data points
    :return: data, labels
    """
    x = torch.empty(n, 2)
    x = x.uniform_(0, 1)

    x_centered = x - 0.5
    norm_squared = x_centered.pow(2).sum(dim=1)
    radius_sq = 1 / (2 * math.pi)

    # To check if the points are inside the disk
    y = norm_squared.sub(radius_sq).sign().add(1).div(2)
    return x, y


def encode_labels(target):
    """
    Encode the labels:
    if 1 -> [0,1]
    if 0 -> [0,0]
    """
    encoded = torch.empty(target.shape[0], 2)
    encoded[:, 0].fill_(0)
    encoded[:, 1] = target
    return encoded


def split_data(X, y, train_ratio=0.7):

    validation_ratio = 1 - train_ratio
    train_size = math.floor(X.size()[0] * train_ratio)
    validation_size = math.floor(X.size()[0] * validation_ratio)

    train_inputs = X.narrow(0, 0, train_size)
    train_targets = y.narrow(0, 0, train_size)

    validation_inputs = X.narrow(0, train_size, validation_size)
    validation_targets = y.narrow(0, train_size, validation_size)

    return train_inputs, train_targets, validation_inputs, validation_targets


def train_model(model, X_train, y_train, X_val, y_val, epochs=150, batch_size=1, loss="MSE", lr=0.01,
                verbose=True):
    train_loss = []
    train_acc = []

    val_loss = []
    val_acc = []

    optimizer = SGD(model=model, lr=lr)

    criterion = MSE(model=model)

    number_batches = X_train.size(0) // batch_size

    for epoch in range(epochs):

        epoch_train_loss = 0
        epoch_train_acc = 0

        epoch_val_loss = 0
        epoch_val_acc = 0

        # Train
        for index in range(0, X_train.size(0), batch_size):
            X_train_batch = X_train[index:(index + batch_size)]
            y_train_batch = y_train[index:(index + batch_size)]
            encoded_y_train_batch = encode_labels(y_train_batch)

            # forward pass
            output = model.forward(X_train_batch)

            loss = criterion.forward(output, encoded_y_train_batch)
            epoch_train_loss += loss.item()

            batch_acc = (output.max(1)[1].float() == y_train_batch).sum().item()
            epoch_train_acc += batch_acc

            # Set gradients of all model parameters to zero.
            optimizer.zero_grad()
            # backward pass and update parameters gradient
            criterion.backward()
            # update parameters (SGD step)
            optimizer.step()

        epoch_train_acc = (epoch_train_acc / X_train.size(0)) * 100
        epoch_train_loss = (epoch_train_loss / number_batches)

        train_acc.append(epoch_train_acc)
        train_loss.append(epoch_train_loss)

        # Validation
        for index in range(0, X_val.size(0), batch_size):
            X_val_batch = X_val[index:(index + batch_size)]

            y_val_batch = y_val[index:(index + batch_size)]
            encoded_y_val_batch = encode_labels(y_val_batch)

            # forward pass
            output = model.forward(X_val_batch)
            loss = criterion.forward(output, encoded_y_val_batch)
            epoch_val_loss += loss.item()

            batch_acc = (output.max(1)[1].float() == y_val_batch).sum().item()
            epoch_val_acc += batch_acc

        epoch_val_acc = (epoch_val_acc / X_val.size(0)) * 100
        epoch_val_loss = (epoch_val_loss / number_batches)

        val_acc.append(epoch_val_acc)
        val_loss.append(epoch_val_loss)

        if verbose and ( ((epoch + 1) % 5 == 0) or (epoch == 0) ):
            print(f"Epoch {epoch + 1}: train loss={epoch_train_loss:0.4f}, train acccuracy={epoch_train_acc:0.2f}% | " + \
                  f"validation loss={epoch_val_loss:0.4f}, validation acccuracy={epoch_val_acc:0.2f}%")

    return model, train_loss, train_acc, val_loss, val_acc


def test_model(trained_model, X_test, y_test, batch_size=1):
    
    test_acc = 0
    test_loss = 0

    criterion = MSE(model=trained_model)
    
    for index in range(0, X_test.size(0), batch_size):
        
        X_test_batch = X_test[index:(index + batch_size)]

        y_test_batch = y_test[index:(index + batch_size)]
        encoded_y_test_batch = encode_labels(y_test_batch)

        # forward pass
        output = trained_model.forward(X_test_batch)
        loss = criterion.forward(output, encoded_y_test_batch)
        test_loss += loss.item()

        acc = (output.max(1)[1].float() == y_test_batch).sum().item()
        test_acc += acc

    test_acc = (test_acc / X_test.size(0)) * 100
    test_loss = (test_loss / X_test.size(0))
    
    print(f"Test loss={test_loss:0.4f}, test accuracy={test_acc:0.2f}")

    return test_loss, test_acc


# Testing the model by each epoch
def train_test_model(model, X_train, y_train, X_test, y_test, epochs=150, batch_size=1, loss="MSE", lr=0.01,
                     verbose=True):
    train_loss = []
    train_acc = []

    test_loss = []
    test_acc = []

    optimizer = SGD(model=model, lr=lr)

    criterion = MSE(model=model)

    number_batches = X_train.size(0) // batch_size

    for epoch in range(epochs):

        epoch_train_loss = 0
        epoch_train_acc = 0

        epoch_test_loss = 0
        epoch_test_acc = 0

        # Train
        for index in range(0, X_train.size(0), batch_size):
            X_train_batch = X_train[index:(index + batch_size)]
            y_train_batch = y_train[index:(index + batch_size)]
            encoded_y_train_batch = encode_labels(y_train_batch)

            # forward pass
            output = model.forward(X_train_batch)

            loss = criterion.forward(output, encoded_y_train_batch)
            epoch_train_loss += loss.item()

            batch_acc = (output.max(1)[1].float() == y_train_batch).sum().item()
            epoch_train_acc += batch_acc

            # Set gradients of all model parameters to zero.
            optimizer.zero_grad()
            # backward pass and update parameters gradient
            criterion.backward()
            # update parameters (SGD step)
            optimizer.step()

        epoch_train_acc = (epoch_train_acc / X_train.size(0)) * 100
        epoch_train_loss = (epoch_train_loss / number_batches)

        train_acc.append(epoch_train_acc)
        train_loss.append(epoch_train_loss)

        # Test        
        for index in range(0, X_test.size(0), batch_size):
            X_test_batch = X_test[index:(index + batch_size)]

            y_test_batch = y_test[index:(index + batch_size)]
            encoded_y_test_batch = encode_labels(y_test_batch)

            # forward pass
            output = model.forward(X_test_batch)
            loss = criterion.forward(output, encoded_y_test_batch)
            epoch_test_loss += loss.item()

            batch_acc = (output.max(1)[1].float() == y_test_batch).sum().item()
            epoch_test_acc += batch_acc

        epoch_test_acc = (epoch_test_acc / X_test.size(0)) * 100
        epoch_test_loss = (epoch_test_loss / number_batches)

        test_acc.append(epoch_test_acc)
        test_loss.append(epoch_test_loss)

        if verbose and (((epoch + 1) % 5 == 0) or (epoch == 0)):
            print(f"Epoch {epoch + 1}: train loss={epoch_train_loss:0.4f}, train acccuracy={epoch_train_acc:0.2f}% | " + \
                  f"test loss={epoch_test_loss:0.4f}, test acccuracy={epoch_test_acc:0.2f}%")

    return model, train_loss, train_acc, test_loss, test_acc


## PyTorch
def torch_train_test_model(model, X_train, y_train, X_test, y_test, epochs=150, batch_size=1, loss="MSE", lr=0.01,
                           verbose=True):
    train_loss = []
    train_acc = []

    test_loss = []
    test_acc = []

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    criterion = torch.nn.MSELoss()

    number_batches = X_train.size(0) // batch_size

    for epoch in range(epochs):

        epoch_train_loss = 0
        epoch_train_acc = 0

        epoch_test_loss = 0
        epoch_test_acc = 0

        # Train
        for index in range(0, X_train.size(0), batch_size):
            X_train_batch = X_train[index:(index + batch_size)]
            y_train_batch = y_train[index:(index + batch_size)]
            encoded_y_train_batch = encode_labels(y_train_batch)

            # forward + backward + optimize
            outputs = model(X_train_batch)
            loss = criterion(outputs, encoded_y_train_batch)

            # Set gradients of all model parameters to zero.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            batch_acc = (outputs.max(1)[1].float() == y_train_batch).sum().item()
            epoch_train_acc += batch_acc

        epoch_train_acc = (epoch_train_acc / X_train.size(0)) * 100
        epoch_train_loss = (epoch_train_loss / number_batches)

        train_acc.append(epoch_train_acc)
        train_loss.append(epoch_train_loss)

        # Test        
        for index in range(0, X_test.size(0), batch_size):
            X_test_batch = X_test[index:(index + batch_size)]
            y_test_batch = y_test[index:(index + batch_size)]
            encoded_y_test_batch = encode_labels(y_test_batch)

            # forward pass
            outputs = model(X_test_batch)
            loss = criterion(outputs, encoded_y_test_batch)
            epoch_test_loss += loss.item()

            batch_acc = (outputs.max(1)[1].float() == y_test_batch).sum().item()
            epoch_test_acc += batch_acc

        epoch_test_acc = (epoch_test_acc / X_test.size(0)) * 100
        epoch_test_loss = (epoch_test_loss / number_batches)

        test_acc.append(epoch_test_acc)
        test_loss.append(epoch_test_loss)

        if verbose and (((epoch + 1) % 5 == 0) or (epoch == 0)):
            print(f"Epoch {epoch + 1}: train loss={epoch_train_loss:0.4f}, train acccuracy={epoch_train_acc:0.2f}% | " + \
                  f"test loss={epoch_test_loss:0.4f}, test acccuracy={epoch_test_acc:0.2f}%")

    return model, train_loss, train_acc, test_loss, test_acc
