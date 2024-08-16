import pygame
import numpy
import time
pygame.init()

SCREENWIDTH = 1000
SCREENHEIGHT = 800
screen = pygame.display.set_mode([SCREENWIDTH, SCREENHEIGHT]) #Creates display for user
clock = pygame.time.Clock()
FPS = 60
global G
global mass
G = 6.67430 * 10**-11 #Universal Gravitational Constant
mass = 5.972 * 10**24 #Mass of Earth
pi = numpy.pi
zoomLevel = 100
font = pygame.font.SysFont("Verdana", 22)
smallFont = pygame.font.SysFont("Verdana", 15)
steps = 10
phi = pi/2
theta = 0
state = numpy.array([0, 0, 1.03139224611, -0.00122429759, 0, 0])
#state = numpy.array([0, 0, 1.15696123058, -0.00120682781, 0.0003, 0.0001])
burnVector = numpy.array([0, 0, 0])
burnChoice = "stability assist"
perigeeCoordinate = [0, 0, 0]
apogeeCoordinate = [0, 0, 0]
thrust = 0
iteration_timeWarp = 1
vector_timeWarp = 1
deltaV = 0
rightClick = False
run = True
booted = False
WHITE = (255, 255, 255)#(0, 0, 0)
BLACK = (0, 0, 0)#(255, 255, 255)

#--------- Sub programs -----------

def findGravity(state):
    #Calculation for gravity is a = G*M2/h^2 (G = constant, M2 = mass of earth, h = altitude above centre of earth)
    #For the purpose of this program, we can ignore the gravity affecting the Earth from the rocket
    gravitationalAcceleration = (G * mass) / ((numpy.linalg.norm(state[:3])*6371000) ** 2)/6371000
    unitGravityVector = -state[:3] / numpy.linalg.norm(state[:3])
    gravityVector = unitGravityVector * gravitationalAcceleration
    return gravityVector

def orthographicProjection(colour, coordinate, radius):
    projected_coordinate = (coordinate[0]+SCREENWIDTH//2, coordinate[1]+SCREENHEIGHT//2)
    pygame.draw.circle(screen, colour, projected_coordinate, radius) #Draws point to screen

def RotationX(coordinate, theta):
    rotationXMatrix = [[1, 0, 0], [0, numpy.cos(theta), -numpy.sin(theta)], [0, numpy.sin(theta), numpy.cos(theta)]]
    newCoordinate = numpy.around(numpy.matmul(rotationXMatrix, coordinate), 5)
    return newCoordinate

def RotationY(coordinate, phi):
    rotationYMatrix = [[numpy.cos(phi), 0, numpy.sin(phi)], [0, 1, 0], [-numpy.sin(phi), 0, numpy.cos(phi)]]
    newCoordinate = numpy.around(numpy.matmul(rotationYMatrix, coordinate), 5)
    return newCoordinate

def cameraZoom(coordinate, zoomLevel):
    enlargedCoordinate = coordinate * zoomLevel
    return enlargedCoordinate

def polarCoordinateGenerator(steps):
    point = [0, 1, 0]
    earthList = []
    theta = pi/(steps-1)
    while theta < 3.141: #Pi with tolerance due to python rounding error
        stepSize = pi/(steps-1)
        Xpoint = RotationX(point, theta) #Rotate point around X axis
        theta += stepSize
        phi = 0
        while phi < 2*3.141: #2*pi with tolerance
            stepSize = pi/steps
            Ypoint = RotationY(Xpoint, phi) #Rotate point around Y axis
            phi += stepSize
            earthList.append(Ypoint) #Add rotated point to list
    earthList.append(numpy.array([0, 1, 0]))
    earthList.append(numpy.array([0, -1, 0]))
    return earthList

def earthScale(anyValue):
    scaledValue = anyValue * 6371 #Finds distance in coordinates and converts to real distance in km
    return scaledValue

def update_rocketVelocity(state, gravityVector, burnVector):
    state[3:] += (gravityVector/FPS + numpy.array(burnVector)/FPS) # Adds all forces to velocity
    return state

def update_rocketCoordinate(state):
    state[:3] += (state[3:]/FPS) #Adds velocity to rocketCoordinate every second, spread out over 'FPS' frames
    return state

def derivatives(t, state):
    r, v = state[:3], state[3:] #Splits state vector into position vector (r) and velocity (v)
    drdt = v #Derivative of position in respect to time is velocity
    dvdt = findGravity(r) #Derivative of velocity in respect to time is acceleration
    return numpy.concatenate([drdt, dvdt]) #Return derivative of state

def estimateOrbitalPath(state):
    orbitList = []
    initial = state[:3]
    t = 0 #Initial time
    dt = 50 #Time step in seconds
    steps = 300 #Number of time steps
    isOrbitComplete = False
    orbitList.append(state[:3]) #Adds current rocket coordinate to trajectory
    for count in range(steps):
        v = numpy.linalg.norm(state[3:])
        timePeriod = min((2*pi)*((numpy.linalg.norm(apogeeCoordinate)+numpy.linalg.norm(perigeeCoordinate))**(3/2))/(numpy.sqrt(G*mass)), 0.0001)
        dt = (timePeriod**(2/3))*1000/v/2

        k1 = dt * derivatives(t, state) #Find k1
        k2 = dt * derivatives(t + dt/2, state + k1/2) #Find k2
        k3 = dt * derivatives(t + dt/2, state + k2/2) #Find k3
        k4 = dt * derivatives(t + dt, state + k3) #Find k4
        state1 = state + (k1 + 2*k2 + 2*k3 + k4) / 6 #Apply to state vector

        displacementFromStart = numpy.linalg.norm(state[:3] - initial)
        if count > 10 and displacementFromStart < (numpy.linalg.norm(state1[:3]-state[:3])): #If the calculated point is near the red circle, end the loop
            isOrbitComplete = True #Flag for checking if orbit is complete (initially False)
            break
        state = state1
        t = t + dt #Increment time
        orbitList.append(state.copy())
    return orbitList, isOrbitComplete

def findValues(orbitList, state):
    inclinationList = []
    count = 0
    perigeeCoordinate = state[:3]
    apogeeCoordinate = state[:3]
    perigeeAltitude = 100000
    apogeeAltitude = 0
    while count < len(orbitList)-1:
        previousCoordinate = orbitList[count]
        previousAltitude = numpy.linalg.norm(previousCoordinate)
        count += 1
        currentCoordinate = orbitList[count]
        currentAltitude = numpy.linalg.norm(currentCoordinate)

        if 0.1 > currentCoordinate[1] and currentCoordinate[1] > -0.1:
            inclinationList.append(currentCoordinate)
        if currentAltitude > apogeeAltitude:
            apogeeCoordinate = currentCoordinate[:3]
            apogeeAltitude = numpy.linalg.norm(currentCoordinate)
        elif currentAltitude < perigeeAltitude:
            perigeeCoordinate = currentCoordinate[:3]
            perigeeAltitude = numpy.linalg.norm(currentCoordinate)
    
    count = 0
    ascendingNode = numpy.array([0, 0, 0])
    descendingNode = numpy.array([0, 0, 0])
    while count < len(orbitList)-1:
        inclinationCoord0 = orbitList[count]
        inclinationCoord1 = orbitList[count+1]
        isPlaneCrossed = inclinationCoord0[1] * inclinationCoord1[1]
        
        if isPlaneCrossed < 0:
            inclinationVector = inclinationCoord1[3:]
            if inclinationVector[1] > 0:
                ascendingNode = inclinationCoord1[:3]
            else:
                descendingNode = inclinationCoord1[:3]
        count += 1
    if numpy.linalg.norm(ascendingNode) == 0:
        ascendingNode = state[:3]
    if numpy.linalg.norm(descendingNode) == 0:
        descendingNode = state[:3]
    if perigeeCoordinate[1] == 0 and apogeeCoordinate[1] == 0:
        ascendingNode = apogeeCoordinate
        descendingNode = perigeeCoordinate
        
    return apogeeCoordinate, perigeeCoordinate, ascendingNode, descendingNode

def cameraZoomLevel(mouseScroll, currentZoomLevel):
    #mouseScroll outputs negative value for scroll down, positive value for scroll up
    zoomLevel = currentZoomLevel + 3*mouseScroll #value from mouseScroll is added to zoomLevel
    return zoomLevel

def rotationAmount(mouseX, mouseY, previous_mouseX, previous_mouseY, theta, phi):
    theta += -(mouseX - previous_mouseX) * 0.008 #Update theta by difference in mouseX
    phi += (mouseY - previous_mouseY) * 0.008 #Update phi by difference in mouseY
    phi = max(min(phi, pi/2), -pi/2) #Ensure phi is locked to +-pi/2 to stop users accidentally being disoriented
    return theta, phi

def GUI(burnChoice, thrust):
    #Setting colours for buttons

    offColour = (255, 158, 158)
    onColour = (158, 255, 158)
    
    progradeColour = offColour
    retrogradeColour = offColour
    radialinColour = offColour
    radialoutColour = offColour
    normalColour = offColour
    antinormalColour = offColour

    if burnChoice == "prograde":
        progradeColour = onColour
    elif burnChoice == "retrograde":
        retrogradeColour = onColour
    elif burnChoice == "radialin":
        radialinColour = onColour
    elif burnChoice == "radialout":
        radialoutColour = onColour
    elif burnChoice == "normal":
        normalColour = onColour
    elif burnChoice == "antinormal":
        antinormalColour = onColour

    #Draw prograde button
    pygame.draw.circle(screen, BLACK, (75, 75), 30) #Base circle
    pygame.draw.circle(screen, progradeColour, (75, 75), 25)
    pygame.draw.circle(screen, BLACK, (75, 75), 15)
    pygame.draw.line(screen, BLACK, (96, 75), (53, 75), 5) #3 lines protruding from circle
    pygame.draw.line(screen, BLACK, (75, 75), (75, 54), 5)
    pygame.draw.circle(screen, progradeColour, (75, 75), 10) #Creating gap in middle
    pygame.draw.circle(screen, BLACK, (75, 75), 3) #Placing central black dot

    #Draw retrograde button
    pygame.draw.circle(screen, BLACK, (150, 75), 30) #Base circle
    pygame.draw.circle(screen, retrogradeColour, (150, 75), 25)
    pygame.draw.circle(screen, BLACK, (150, 75), 15)
    pygame.draw.line(screen, BLACK, (150, 75), (150, 52), 5) #3 lines protruding from circle
    pygame.draw.line(screen, BLACK, (130, 85), (150, 75), 5)
    pygame.draw.line(screen, BLACK, (169, 85), (150, 75), 5)
    pygame.draw.circle(screen, retrogradeColour, (150, 75), 10) #Create gap in middle
    pygame.draw.line(screen, BLACK, (158, 83), (142, 67), 5) #2 diagonal lines to fill gap
    pygame.draw.line(screen, BLACK, (141, 83), (157, 67), 5)
    pygame.draw.circle(screen, retrogradeColour, (150, 75), 25, width=3) #Neaten edges of 3 protruding lines

    #Draw normal button
    pygame.draw.circle(screen, BLACK, (75, 150), 30) #Base circle
    pygame.draw.circle(screen, normalColour, (75, 150), 25)
    pygame.draw.line(screen, BLACK, (92, 160), (58, 160), 5) #Triangle
    pygame.draw.line(screen, BLACK, (92, 160), (75, 130), 5)
    pygame.draw.line(screen, BLACK, (58, 160), (75, 130), 5)
    pygame.draw.circle(screen, BLACK, (75, 150), 3) #Middle circle

    #Draw antinormal button
    pygame.draw.circle(screen, BLACK, (150, 150), 30) #Base circle
    pygame.draw.circle(screen, antinormalColour, (150, 150), 25)
    pygame.draw.line(screen, BLACK, (150, 170), (167, 140), 5) #Upside down triangle
    pygame.draw.line(screen, BLACK, (150, 170), (133, 140), 5)
    pygame.draw.line(screen, BLACK, (167, 140), (133, 140), 5)
    pygame.draw.line(screen, BLACK, (150, 150), (150, 128), 5) #3 lines
    pygame.draw.line(screen, BLACK, (150, 150), (169, 161), 5)
    pygame.draw.line(screen, BLACK, (150, 150), (130, 161), 5)
    pygame.draw.circle(screen, antinormalColour, (152, 150), 7) #Covering up overlap from 3 lines
    pygame.draw.circle(screen, antinormalColour, (149, 150), 7)
    pygame.draw.circle(screen, BLACK, (150, 150), 3) #Middle circle
    pygame.draw.circle(screen, antinormalColour, (150, 150), 25, width=3) #Neaten up edges from 3 lines

    #Draw radialin button
    pygame.draw.circle(screen, BLACK, (75, 225), 30) #Base circle
    pygame.draw.circle(screen, radialinColour, (75, 225), 25)
    pygame.draw.circle(screen, BLACK, (75, 225), 15)
    pygame.draw.circle(screen, radialinColour, (75, 225), 10)
    pygame.draw.line(screen, BLACK, (83, 233), (67, 217), 5) #2 diagonal lines crossing in centre
    pygame.draw.line(screen, BLACK, (66, 233), (82, 217), 5)
    pygame.draw.circle(screen, radialinColour, (75, 225), 5) #Cover up middle of crossing lines
    
    #Draw radialout button
    pygame.draw.circle(screen, BLACK, (150, 225), 30) #Base circle
    pygame.draw.circle(screen, radialoutColour, (150, 225), 25)
    pygame.draw.circle(screen, BLACK, (150, 225), 15)
    pygame.draw.line(screen, BLACK, (166, 241), (134, 209), 5) #Draw protruding lines
    pygame.draw.line(screen, BLACK, (133, 241), (165, 209), 5)
    pygame.draw.circle(screen, radialoutColour, (150, 225), 10) #Cover middle overlap
    pygame.draw.circle(screen, radialoutColour, (150, 225), 25, width=5) #Neaten edges of protruding lines
    pygame.draw.circle(screen, BLACK, (150, 225), 3) #Add black dot to centre

    #Draw Thrust Bar (thrust should be passed as a %, between 1-100)
    pygame.draw.rect(screen, BLACK, (40, 295, 145, 30))
    pygame.draw.rect(screen, "green", (45, 300, thrust*135/100, 20))

def burnChoices(current_burnChoice, mouseX, mouseY):

    #Set locations of all points to compare
    mouseLocation = numpy.array([mouseX, mouseY])
    progradeLocation = numpy.array([75, 75])
    retrogradeLocation = numpy.array([150, 75])
    normalLocation = numpy.array([75, 150])
    antinormalLocation = numpy.array([150, 150])
    radialinLocation = numpy.array([75, 225])
    radialoutLocation = numpy.array([150, 225])

    #Checks if mouse location is near a button and sets burnchoice accordingly
    if numpy.linalg.norm(mouseLocation-progradeLocation) < 40:
        burnChoice = "prograde"
    elif numpy.linalg.norm(mouseLocation-retrogradeLocation) < 40:
        burnChoice = "retrograde"
    elif numpy.linalg.norm(mouseLocation-normalLocation) < 40:
        burnChoice = "normal"
    elif numpy.linalg.norm(mouseLocation-antinormalLocation) < 40:
        burnChoice = "antinormal"
    elif numpy.linalg.norm(mouseLocation-radialinLocation) < 40:
        burnChoice = "radialin"
    elif numpy.linalg.norm(mouseLocation-radialoutLocation) < 40:
        burnChoice = "radialout"
    else:
        burnChoice = current_burnChoice
    return burnChoice

def orbitalPlane(state):
    planeVector = -numpy.cross(state[3:], state[:3]) #Finds vector perpendicular to coordinate and velocity
    unitPlaneVector = planeVector / numpy.linalg.norm(planeVector) #Creates unit vector
    return unitPlaneVector

def findProgradeVector(state, thrust):
    progradeVector = state[3:]/numpy.linalg.norm(state[3:]) #Creates unit vector in direction of velocity
    progradeVector *= thrust #Multiplies by thrust
    return progradeVector

def findRetrogradeVector(state, thrust):
    retrogradeVector = -state[3:]/numpy.linalg.norm(state[3:]) #Creates unit vector in opposite direction of velocity
    retrogradeVector *= thrust #Multiplies by thrust
    return retrogradeVector

def findNormalVector(planeVector, thrust):
    normalVector = planeVector * thrust #Multiplies unit plane vector by thrust
    return normalVector

def findAntinormalVector(planeVector, thrust):
    antinormalVector = -planeVector * thrust #Multiplies negative unit plane vector by thrust
    return antinormalVector

def findRadialinVector(state, planeVector, thrust):
    rocketCoordinate = state[:3] #Assign rocket coordinate
    direction_radialinVector = -rocketCoordinate #Find direction of radial in vector
    unit_radialinVector = direction_radialinVector/numpy.linalg.norm(direction_radialinVector) #Create unit vector
    radialinVector = unit_radialinVector * thrust #Multiply by thrust
    return radialinVector

def findRadialoutVector(state, planeVector, thrust):
    rocketCoordinate = state[:3] #Assign rocket coordinate
    direction_radialoutVector = rocketCoordinate #Find direction of radial out vector
    unit_radialoutVector = direction_radialoutVector/numpy.linalg.norm(direction_radialoutVector) #Create unit vector
    radialoutVector = unit_radialoutVector * thrust #Multiply by thrust
    return radialoutVector

def findBurnVector(current_burnChoice, current_burnVector, state, planeVector, thrust):
    if current_burnChoice == "prograde": #If burn choice is prograde
        burnVector = findProgradeVector(state, thrust) #Set burn vector to prograded
    elif current_burnChoice == "retrograde": #Etc.
        burnVector = findRetrogradeVector(state, thrust)
    elif current_burnChoice == "normal":
        burnVector = findNormalVector(planeVector, thrust)
    elif current_burnChoice == "antinormal":
        burnVector = findAntinormalVector(planeVector, thrust)
    elif current_burnChoice == "radialin":
        burnVector = findRadialinVector(state, planeVector, thrust)
    elif current_burnChoice == "radialout":
        burnVector = findRadialoutVector(state, planeVector, thrust)
    else:
        burnVector = current_burnVector #Dont change burn vector if no burn choice is selected (stays [0, 0, 0])
    return burnVector

def editThrust(currentThrust, keyPressed):
    if keyPressed == 'z':
        thrust = 0.00000304793 #Set thrust to max
    elif keyPressed == 'x':
        thrust = 0             #Set thrust to 0
    elif keyPressed == 'shift':
        thrust = currentThrust + 0.00000304793/FPS #Increment thrust
    elif keyPressed == 'ctrl':
        thrust = currentThrust - 0.00000304793/FPS #Decrement thrust
    else:
        thrust = currentThrust #No change
    return thrust

def timeWarp(current_iterationTimeWarp, current_vectorTimeWarp, keyPressed):
    #Time warp values: 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000
    iteration_timeWarp = current_iterationTimeWarp #in case: iterationTW = 0 or 1000
    vector_timeWarp = current_vectorTimeWarp #in case: vectorTW = 0 or 100
    #INCREASING TIME WARP
    if keyPressed == "period":
        #VECTOR TIME WARP
        if str(current_vectorTimeWarp)[:1] == "1" and current_vectorTimeWarp < 100:
            vector_timeWarp = current_vectorTimeWarp * 5 #5, 50
        elif str(current_vectorTimeWarp)[:1] == "5" and current_vectorTimeWarp < 100:
            vector_timeWarp = current_vectorTimeWarp * 2 #10, 100
        #ITERATION TIME WAP
        elif str(current_iterationTimeWarp)[:1] == "1" and current_iterationTimeWarp < 100:
            iteration_timeWarp = current_iterationTimeWarp * 5 #5, 50, 500
        elif str(current_iterationTimeWarp)[:1] == "5" and current_iterationTimeWarp < 100:
            iteration_timeWarp = current_iterationTimeWarp * 2 #10, 100, 1000
    #DECREASING TIME WARP
    if keyPressed == "comma":
        #VECTOR TIME WARP
        if str(current_iterationTimeWarp)[:1] == "1" and current_iterationTimeWarp > 1:
            iteration_timeWarp = current_iterationTimeWarp // 2 #50, 5
        elif str(current_iterationTimeWarp)[:1] == "5" and current_iterationTimeWarp > 1:
            iteration_timeWarp = current_iterationTimeWarp // 5 #10, 1
        #ITERATION TIME WARP
        elif str(current_vectorTimeWarp)[:1] == "1" and current_vectorTimeWarp > 1:
            vector_timeWarp = current_vectorTimeWarp // 2 #500, 50, 5
        elif str(current_vectorTimeWarp)[:1] == "5" and current_vectorTimeWarp > 1:
            vector_timeWarp = current_vectorTimeWarp // 5 #100, 10, 1
    return iteration_timeWarp, vector_timeWarp


def draw_text(text, font, colour, x, y):
    text_image = font.render(text, True, colour) #Creates text image
    screen.blit(text_image, (x, y)) #Blits to screen

def extras(isOrbitComplete, state, apogeeCoordinate, perigeeCoordinate, deltaV, iteration_timeWarp, vector_timeWarp):
    #---------- PERIGEE AND APOGEE ----------
    perigee = int(numpy.floor(earthScale(numpy.linalg.norm(perigeeCoordinate))-6371))
    apogee = int(numpy.floor(earthScale(numpy.linalg.norm(apogeeCoordinate))-6371))
    if perigee < 160:
        colour = "red" #display perigee as red if inside of Earth's atmosphere (160km)
    else:
        colour = BLACK #display black otherwise
    draw_text(f"perigee: {perigee}km", font, colour, SCREENWIDTH-300, 0)

    if apogee > 1500000: #If escape velocity is reached, or is almost reached,
        draw_text(f"apogee: inf km", font, "red", SCREENWIDTH-300, 20) #Display 'inf' as apogee should be infinite when escape velocity is reached
    else:
        draw_text(f"apogee: {apogee}km", font, BLACK, SCREENWIDTH-300, 20) #Draw apogee value to screen
    
    #---------- ECCENTRICITY ----------
    if apogee > 1500000:
        draw_text(f"eccentricity: > 1", font, BLACK, SCREENWIDTH-300, 40) #Display eccentricity as >1 as escape velocity is likely surpassed
    elif isOrbitComplete == False:
        draw_text(f"eccentricity: 1", font, BLACK, SCREENWIDTH-300, 40) #Escape velocity is defined as when orbital path does not return (a close approximation)
    else:
        eccentricity = round(1-(2/(numpy.linalg.norm(apogeeCoordinate)/numpy.linalg.norm(perigeeCoordinate)+1)), 2) #Calculate eccentricity to 2 d.p.
        draw_text(f"eccentricity: {eccentricity}", font, BLACK, SCREENWIDTH-300, 40) #Display eccentricity
    
    #---------- ALTITUDE ----------
    altitude = int(numpy.floor(earthScale(numpy.linalg.norm(state[:3]))-6371)) #Calculate altitude
    draw_text(f"altitude: {altitude}km", font, BLACK, SCREENWIDTH-300, 60)

    #---------- VELOCITY ----------
    velocity = numpy.round(numpy.linalg.norm(earthScale(state[3:])), 2) #Calculate velocity
    draw_text(f"Velocity: {velocity}km/s", font, BLACK, SCREENWIDTH-300, 80)

    #---------- INCLINATION ----------
    orbitalInclination = round(numpy.arccos(numpy.dot(orbitalPlane(state), numpy.array([0, 1, 0]))), 2) #Calculate orbital inclination
    if orbitalInclination > pi/2:
        orbitalInclination = round(orbitalInclination-pi, 2) #Inclination is in range 0 to pi, I want it in range -pi/2 to pi/2
    draw_text(f"Inclination: {orbitalInclination} radians", font, BLACK, SCREENWIDTH-300, 100)

    #---------- DELTA V ----------
    draw_text(f"Delta V: {round(deltaV,2)} km/s", font, BLACK, SCREENWIDTH-300, 120) #Display Delta V value

    #---------- TIME WARP ----------
    timeWarpDisplay = iteration_timeWarp * vector_timeWarp #Calculate total time warp
    draw_text(f"Time Warp: {timeWarpDisplay}x", font, BLACK, SCREENWIDTH-300, SCREENHEIGHT-60)

def valuesIcon(Coordinate, text, colour, flag, text2):
    radius = 6
    x, y = Coordinate[0]+SCREENWIDTH//2, Coordinate[1]+SCREENHEIGHT//2
    pygame.draw.polygon(screen, colour, ((x-radius, y-radius), (x, y), (x+radius, y-radius), (x+2*radius, y-radius), (x+2*radius, y-3.25*radius), (x-2*radius, y-3.25*radius), (x-2*radius, y-radius)))
    draw_text(text, smallFont, "black", x-1.75*radius, y-4*radius)
    if flag == True:
        draw_text(text2, font, "black", x-22*radius, y-8*radius)

# -------- Main Program -----------
earthList = polarCoordinateGenerator(steps)
orbitList, isOrbitComplete = estimateOrbitalPath(state) #Create trajectory
apogeeCoordinate, perigeeCoordinate, ascendingNode, descendingNode = findValues(orbitList, state)
while run:
    # ---------- CONTROLS (EVENT LOOP) ----------
    events = pygame.event.get()
    for event in events: 
        if event.type == pygame.QUIT: 
            run = False
        if event.type == pygame.KEYDOWN:
            # ---------- GRAPHICS SETTINGS ----------
            if event.key == pygame.K_DOWN: #IF DOWN ARROW KEY PRESSED
                steps -= 1
                earthList = polarCoordinateGenerator(steps)
            elif event.key == pygame.K_UP: #IF UP ARROW KEY PRESSED
                steps += 1
                earthList = polarCoordinateGenerator(steps)
            # ---------- THRUST MIN/MAX---------- 
            elif event.key == pygame.K_z: #Max thrust
                thrust = editThrust(thrust, "z")
            elif event.key == pygame.K_x: #No thrust
                thrust = editThrust(thrust, "x")
            # ---------- TIME WARP ----------
            elif event.key == pygame.K_PERIOD: #Increase time warp
                iteration_timeWarp, vector_timeWarp = timeWarp(iteration_timeWarp, vector_timeWarp, "period")
            elif event.key == pygame.K_COMMA: #Decrease time warp
                iteration_timeWarp, vector_timeWarp = timeWarp(iteration_timeWarp, vector_timeWarp, "comma")
        # ---------- CAMERA ZOOM ----------
        if event.type == pygame.MOUSEWHEEL:
            mouseScroll = event.y #Detect scroll up/down
            zoomLevel = cameraZoomLevel(mouseScroll, zoomLevel)
        # ---------- BURN VECTOR ----------
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == pygame.BUTTON_LEFT:
                    mouseX, mouseY = pygame.mouse.get_pos()
                    burnChoice = burnChoices(burnChoice, mouseX, mouseY) #Select burn vector with mouse
                    burnVector = findBurnVector(burnChoice, burnVector, state, orbitalPlane(state), thrust) #Apply burn vector
        # ---------- CAMERA ROTATION ----------
            elif event.button == pygame.BUTTON_RIGHT: 
                rightClick = True #Begin rotation script
                previous_mouseX, previous_mouseY = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == pygame.BUTTON_RIGHT: 
                rightClick = False #Stop rotation script

    if rightClick: #Rotation script
        mouseX, mouseY = pygame.mouse.get_pos() #Get current mouse position
        theta, phi = rotationAmount(mouseX, mouseY, previous_mouseX, previous_mouseY, theta, phi) #update theta and phi
        previous_mouseX, previous_mouseY = mouseX, mouseY #update last mouse position

    # ---------- THRUST INCREMENT/DECREMENT ----------
    key = pygame.key.get_pressed() #Get all keys pressed each frame
    if key[pygame.K_LSHIFT]: 
        thrust = min(max(editThrust(thrust, "shift"), 0), 0.00000304793) #Increment thrust
    elif key[pygame.K_LCTRL]: 
        thrust = min(max(editThrust(thrust, "ctrl"), 0), 0.00000304793) #decrement thrust


    # ---------- GAME LOGIC ----------
    clock.tick(FPS)

    #---------- NEXT ROCKET COORDINATE ----------
    burnVector = findBurnVector(burnChoice, burnVector, state, orbitalPlane(state), thrust) #Apply burn vector
    for coordinateSteps in range(iteration_timeWarp):
        gravityVector = findGravity(state) #Find the gravity vector

        state = update_rocketVelocity(state, gravityVector * vector_timeWarp, burnVector * vector_timeWarp) #Find the next velocity
        state = update_rocketCoordinate(numpy.concatenate([state[:3], state[3:]*vector_timeWarp])) #Add multiplied velocity to current coordinate
        state[3:] /= vector_timeWarp #Divide velocity for next iteration
    deltaV += thrust
    #---------- ORBITAL TRAJECTORY ----------
    if thrust > 0 or booted == False:
        orbitList, isOrbitComplete = estimateOrbitalPath(state) #Create trajectory
        apogeeCoordinate, perigeeCoordinate, ascendingNode, descendingNode = findValues(orbitList, state)
    
    #---------- CAMERA TRANSFORMATIONS ----------
    #EARTH
    transformed_earthList = []
    for earthCoordinate in earthList:
        transformed_earthCoordinate = RotationX(RotationY(earthCoordinate, theta), phi) #Apply X and Y rotations to earthList
        if transformed_earthCoordinate[2] < 0: #If the coordinates should be visible, add to a list
            transformed_earthList.append(transformed_earthCoordinate)

    #ROCKET
    transformed_rocket = RotationX(RotationY(cameraZoom(state[:3], zoomLevel), theta), phi)

    #VALUES
    transformed_apogeeCoordinate = RotationX(RotationY(cameraZoom(apogeeCoordinate, zoomLevel), theta), phi)
    transformed_perigeeCoordinate = RotationX(RotationY(cameraZoom(perigeeCoordinate, zoomLevel), theta), phi)
    transformed_ascendingNode = RotationX(RotationY(cameraZoom(ascendingNode, zoomLevel), theta), phi)
    transformed_descendingNode = RotationX(RotationY(cameraZoom(descendingNode, zoomLevel), theta), phi)

    #ORBITLIST
    transformed_trajectoryBack = []
    transformed_trajectoryFront = []
    for trajectoryCoordinate in orbitList:
        transformed_trajectoryCoordinate = RotationX(RotationY(cameraZoom(trajectoryCoordinate[:3], zoomLevel), theta), phi)
        if transformed_trajectoryCoordinate[2] > 0:
            transformed_trajectoryBack.append(transformed_trajectoryCoordinate)
        else:
            transformed_trajectoryFront.append(transformed_trajectoryCoordinate)
    #INCLINATION

    # ---------- DRAWING TO SCREEN ----------
    screen.fill(WHITE)

    #TRAJECTORY (CASE: Behind Earth)
    for orbitCoordinate in transformed_trajectoryBack:
        if orbitCoordinate[2] > -750:
            radius = numpy.ceil(min(max(150*zoomLevel/abs(orbitCoordinate[2]+750)/steps, 3), 12))
        else:
            radius = 12
        orthographicProjection("grey", orbitCoordinate, radius)

    #ROCKET (CASE: Behind Earth)
    if transformed_rocket[2] >= 0:
        if transformed_rocket[2] > -750:
            radius = numpy.ceil(min(max(450*zoomLevel/abs(transformed_rocket[2]+750)/steps, 5), 15))
        else:
            radius = 12
        orthographicProjection("red", transformed_rocket, radius)

    #VALUES (CASE: Behind Earth)

    #EARTH
    pygame.draw.circle(screen, BLACK, (SCREENWIDTH//2, SCREENHEIGHT//2), zoomLevel+50/steps)
    pygame.draw.circle(screen, WHITE, (SCREENWIDTH//2, SCREENHEIGHT//2), zoomLevel-50/steps + 100/zoomLevel +1)
    for coordinate in transformed_earthList:
        projected_coordinate = cameraZoom(coordinate, zoomLevel)
        radius = numpy.ceil(min(150*zoomLevel/(projected_coordinate[2]+500)/steps, 10))
        orthographicProjection(BLACK, projected_coordinate, radius)

    #TRAJECTORY (CASE: in front of Earth)
    for orbitCoordinate in transformed_trajectoryFront:
        if orbitCoordinate[2] > -750:
            radius = numpy.ceil(min(max(150*zoomLevel/abs(orbitCoordinate[2]+750)/steps, 3), 12)) #Add perspective
        else:
            radius = 12 #Prevent camera clipping
        orthographicProjection("grey", orbitCoordinate, radius)

    #ROCKET (CASE: in front of Earth)
    if transformed_rocket[2] < 0:
        if transformed_rocket[2] > -750:
            radius = numpy.ceil(min(max(450*zoomLevel/abs(transformed_rocket[2]+750)/steps, 5), 15))
        else:
            radius = 12
        orthographicProjection("red", transformed_rocket, radius)

    #VALUES (CASE: Behind Earth)

    GUI(burnChoice, thrust/0.00000304793*100) #Input thrust as %
    extras(isOrbitComplete, state, apogeeCoordinate, perigeeCoordinate, deltaV, iteration_timeWarp, vector_timeWarp)
    booted = True
    pygame.display.update()
pygame.quit()



