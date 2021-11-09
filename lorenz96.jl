"""
Definition of the Lorenz 96 (40-variable) model
"""
function lorenz96(t, y, F)
    circshift(y, -1) - circshift(y, 2) * circshift(y, 1) - y + F
end
