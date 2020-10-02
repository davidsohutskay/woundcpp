from abaqus import *
from abaqusConstants import *
import __main__
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import os
import sys
import time
import string
import random
import math
import bisect

#######################################################################################
def rectangle(modelName, dimensions, CorticalMaterial, CorticalProps, SubcortMaterial, SubCortProps, meshing, timing, JobName, UMAT, ptype):
#######################################################################################

    HH = dimensions[0]
    LL = dimensions[1]
    h = dimensions[2]
    t = dimensions[3]
    waves = dimensions[4]
    perturbation = dimensions[5]
    lam = CorticalProps[0]
    mu = CorticalProps[1]
    g0 = CorticalProps[2]
    bias = meshing[0]
    ecortex = meshing[1]
    esubcort = meshing[2]
    TotalTime = timing[1]
    InitialStep = timing[0]
    MinStep = timing[2]
    MaxStep = timing[3]
    
    # Create new model
    mdb.Model(name=modelName, modelType=STANDARD_EXPLICIT)    
    s = mdb.models[modelName].ConstrainedSketch(name='__profile__', sheetSize=2.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    
    # Dimension rectangle
    s.rectangle(point1=(0.0, 0.0), point2=(LL, -HH))
    p = mdb.models[modelName].Part(name='Part-1', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p.BaseSolidExtrude(sketch=s, depth=t)
    
    # Partition rectangle
    e = p.edges
    c = p.cells    
    
    e_1 = e.findAt(((0., -HH/2., 0.),), ((LL, -HH/2., t ),))
    p.PartitionEdgeByParam(edges=e_1, parameter=h)
    
    e_2 = e.findAt(((0., -HH/2., t ),), ((LL, -HH/2., 0.),))
    p.PartitionEdgeByParam(edges=e_2, parameter=1-h)
         
    pickedCells = c.findAt(((0.0, -HH/2., t/2), ))
    v, e, d = p.vertices, p.edges, p.datums
    v_br = v.findAt(coordinates=(LL,  -h, 0.0))
    v_fr = v.findAt(coordinates=(LL,  -h, t  ))
    v_fl = v.findAt(coordinates=(0.0, -h, t  ))
    p.PartitionCellByPlaneThreePoints(point1=v_br, point2=v_fr, point3=v_fl, cells=pickedCells)
    
    # Materials
    mdb.models[modelName].Material(name=CorticalMaterial)
    mdb.models[modelName].materials[CorticalMaterial].Depvar(n=10)
    mdb.models[modelName].materials[CorticalMaterial].UserMaterial(unsymm=OFF, mechanicalConstants=CorticalProps)
    
    mdb.models[modelName].Material(name=SubCortMaterial)
    mdb.models[modelName].materials[SubCortMaterial].Depvar(n=10)
    mdb.models[modelName].materials[SubCortMaterial].UserMaterial(unsymm=OFF, mechanicalConstants=SubCortProps)

    # Sections
    c_subcort = c.findAt(((0.0, -HH/2., t), ))
    c_cortex  = c.findAt(((0.0, -h/2.,  t), ))

    mdb.models[modelName].HomogeneousSolidSection(name='Subcortex', material=SubCortMaterial, thickness=None)
    region = p.Set(cells=c_subcort, name='Subcortex')
    p.SectionAssignment(region=region, sectionName='Subcortex', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
    
    mdb.models[modelName].HomogeneousSolidSection(name='Cortex', material=CorticalMaterial, thickness=None)
    region = p.Set(cells=c_cortex, name='Cortex')
    p.SectionAssignment(region=region, sectionName='Cortex', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
        
    # Orientation
    # region = regionToolset.Region(cells=c_subcort)
    # orientation=None
    # mdb.models[modelName].parts['Part-1'].MaterialOrientation(region=region, orientationType=USER, additionalRotationType=ROTATION_NONE, localCsys=None, fieldName='', stackDirection=STACK_3)
            
    # Assembly
    a = mdb.models[modelName].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    a.Instance(name='Part-1-1', part=p, dependent=OFF)

    # Step
    mdb.models[modelName].StaticStep(name='Step-1', previous='Initial', timePeriod=TotalTime, maxNumInc=100000, initialInc=InitialStep, minInc=MinStep, maxInc=MaxStep, nlgeom=ON)
    # mdb.models[modelName].StaticRiksStep(name='Step-1', previous='Initial', maxLPF=30.0, initialArcInc=1.0, minArcInc=0.01, maxArcInc=1e+36, totalArcLength=10.0, matrixStorage=UNSYMMETRIC)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, constraints=ON, connectors=ON, engineeringFeatures=ON, adaptiveMeshConstraints=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, predefinedFields=ON, interactions=OFF, constraints=OFF, engineeringFeatures=OFF)
    mdb.models[modelName].FieldOutputRequest(name='F-Output-3', createStepName='Step-1', variables=('EVOL', 'SDV', 'COORD'))
    # regionDef=mdb.models[modelName].rootAssembly.instances['Part-1-1'].sets['Cortex']
    # mdb.models[modelName].HistoryOutputRequest(name='H-Output-3', createStepName='Step-1', variables=('SDV', ), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)
    # mdb.models[modelName].steps['Step-1'].setValues(stabilizationMagnitude=0.0002, stabilizationMethod=DAMPING_FACTOR, continueDampingFactors=False, adaptiveDampingRatio=None)
    

    # Boundary conditions    
    f = a.instances['Part-1-1'].faces
    
    f_front = f.findAt(((LL/2., -HH/2., t),),((LL/2., -h/2., t),))
    f_back = f.findAt(((LL/2., -HH/2., 0.),),((LL/2., -h/2., 0.),))
    f_bottom = f.findAt(((LL/2., -HH, t/2.),))
    f_sides = f.findAt(((0, -HH/2., t/2.),), ((LL, -HH/2., t/2.),), ((0, -h/2., t/2.),), ((LL, -h/2., t/2.),))

    region = a.Set(faces=f_front, name='front')
    mdb.models[modelName].DisplacementBC(name='front', createStepName='Step-1', region=region, u1=UNSET, u2=UNSET, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)
    
    region = a.Set(faces=f_back, name='back')
    mdb.models[modelName].DisplacementBC(name='back', createStepName='Step-1', region=region, u1=UNSET, u2=UNSET, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)
    
    region = a.Set(faces=f_bottom, name='bottom')
    mdb.models[modelName].PinnedBC(name='bottom', createStepName='Step-1', region=region, localCsys=None)
    
    region = a.Set(faces=f_sides, name='sides')
    mdb.models[modelName].DisplacementBC(name='sides', createStepName='Step-1', region=region, u1=0.0, u2=UNSET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)

    region = a.Set(faces=f, name='all')
    
    # Meshing    
    e = a.instances['Part-1-1'].edges

    e_subcort1 = e.findAt(((LL, -HH/2., t  ), ), ((0.0, -HH/2., 0.0), ))
    e_subcort2 = e.findAt(((LL, -HH/2., 0.0), ), ((0.0, -HH/2., t  ), ))
    a.seedEdgeByBias(biasMethod=SINGLE, end1Edges=e_subcort1, end2Edges=e_subcort2, ratio=bias, number=esubcort, constraint=FINER)
    
    e_cort = e.findAt(((0.0, -h/2, 0.0), ), ((LL, -h/2, 0.0), ), (( 0.0, -h/2, t), ), ((LL, -h/2, t), ))
    a.seedEdgeByNumber(edges=e_cort, number=ecortex, constraint=FINER)
    
    e_width = e.findAt(((LL/2., -h, t), ), ((LL/2., -h, 0.0), ), ((LL/2., 0.0, 0.0), ), ((LL/2., -HH, t), ), ((LL/2, -HH, 0.0), ), ((LL/2, 0.0, t), ))
    a.seedEdgeByNumber(edges=e_width, number=ew, constraint=FINER)

    elemType1 = mesh.ElemType(elemCode=C3D8, elemLibrary=STANDARD, secondOrderAccuracy=OFF, distortionControl=DEFAULT)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
    elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)
    # elemType1 = mesh.ElemType(elemCode=C3D20, elemLibrary=STANDARD)
    # elemType2 = mesh.ElemType(elemCode=C3D15, elemLibrary=STANDARD)
    # elemType3 = mesh.ElemType(elemCode=C3D10, elemLibrary=STANDARD)
    
    c = a.instances['Part-1-1'].cells
    c_all = c.findAt(((LL/2., -h/2., t), ), ((LL/2., -HH/2., t), ))
    pickedRegions =(c_all, )
    a.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, elemType3))
    partInstances =(a.instances['Part-1-1'], )
    a.generateMesh(regions=partInstances)

    # disturb mesh
    e_interface = e.findAt( ((LL/2., -h, t),), ((LL/2., -h, 0.0),), )
    region = a.Set(edges=e_interface, name='interface')
    
    nodes = a.sets['interface'].nodes
    
    coors = []
    
    if ptype == 4:
        wavelengths = []
        for i in range(waves):
            wavelengths.append(random.random()*10.)            
    
    for n in nodes:
        x = n.coordinates[0]
        y = n.coordinates[1]
        z = n.coordinates[2]
        
        if ptype == 0:
            p_n = random.random()*2. - 1.
        elif ptype == 1:
            wavelength = LL/waves
            p_n = -cos(x/wavelength*2.*math.pi)
        elif ptype == 2:
            delta = abs(x - LL/2.)
            wavelength = LL/waves
            if delta <= 0.02525*LL:
                p_n = cos(x/wavelength*2.*math.pi) 
            else: 
                p_n = 0.
        elif ptype == 3:
            p_n = 0.
        elif ptype == 4:
            p_n = 0
            for i in range(waves):
                p_n = p_n + sin(x/wavelengths[i]*2.*math.pi)
    
        y_p = y + perturbation * p_n
        
        coors.append((x, y_p, z))
    
    a.editNode(nodes=nodes, coordinates=coors)    

    # job
    mdb.Job(name=JobName, model=modelName, description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, 
        userSubroutine=UMAT, scratch='', multiprocessingMode=DEFAULT, numDomains=4, numCpus=4, numGPUs=0)
    mdb.jobs[JobName].writeInput(consistencyChecking=OFF)
    

#######################################################################################    
def GetKeywordPosition(blockPrefix):
#######################################################################################    
    mdb.models[modelName].keywordBlock.synchVersions() 
    pos = 0
    positions = []
    for block in mdb.models[modelName].keywordBlock.sieBlocks:
        if string.lower(block[0:len(blockPrefix)])==string.lower(blockPrefix):
            positions.append(pos)
        pos=pos+1
    return positions

#######################################################################################    
def captureimages(JobName,tf):
#######################################################################################
    directory = os.getcwd()
    odb = '.odb'
    sta = '.sta'
    odbName = directory + '/' + JobName + odb
    staName = directory + '/' + JobName + sta
    o1 = session.openOdb(name=odbName)

    viewport = session.viewports['Viewport: 1']
    session.viewports['Viewport: 1'].setValues(displayedObject=o1)
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    
    # save undeformed image with orientations
    session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(ORIENT_ON_UNDEF, ))
    session.viewports['Viewport: 1'].odbDisplay.materialOrientationOptions.setValues(randomSamplingFactor=2.5)
    session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(renderStyle=WIREFRAME, )
    session.printOptions.setValues(vpDecorations=OFF, vpBackground=OFF, compass=OFF, reduceColors=False)
    session.printToFile(fileName=JobName + '-0', format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))
    
    # save final image
    session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(variableLabel='SDV1', outputPosition=INTEGRATION_POINT, )
    session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(minAutoCompute=OFF, minValue=0.5, maxAutoCompute=OFF, maxValue=1.5)    
    session.printToFile(fileName=JobName + '-2', format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))

    
    # collect frame and timestep data
    f = open(staName,'r')
    lines = f.readlines()[5:]
    
    times = []
    frames = []
    
    for line in lines:
        if len(line.split()) >= 7:
            attempt = line.split()[2]
            if 'U' not in attempt:
                frame = int(line.split()[1])
                time = float(line.split()[6])
                frames.append(frame)
                times.append(time)
    frames.append(frames[-1])
    times.append(10)
    
    
    # capture images at each of desired time points
    ts = [tf]
    
    for i in range(len(ts)):
        t = ts[i]
        
        b = bisect.bisect(times,t) # index of closest value greater than desired time
        a = b - 1 # index of closest value less than desired time 
        db = times[b] - t
        da = times[a] - t
        if db <= da: # if tb is closer than ta
            frameNumber = b
        else: 
            frameNumber = a
    
        pictureName = JobName + '-' + str(i+1)
        session.viewports[session.currentViewportName].odbDisplay.setFrame(step='Step-1', frame=frameNumber)
        #session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(minAutoCompute=ON, maxAutoCompute=ON)
        session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(minAutoCompute=OFF, minValue=0.5, maxAutoCompute=OFF, maxValue=1.5)    
        session.printToFile(fileName=pictureName, format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))

#######################################################################################
def writeresults(JobName,f,comptime,lam,mu,g0,alpha,perturbation):
#######################################################################################

    staName = JobName + ".sta"
    g = open(staName, "r")
    tau = g.readlines()[-3].split()[6]
    
    f.write("%s ran until %s \n" %(JobName, tau))

#######################################################################################
if __name__ == "__main__":
#######################################################################################
    
    params = [-2.5, -2, -1.5, -1.]
    T = [30., 20., 5., 2.]    
    T = [35., 20., 7., 3.]

    simulations = [2,3,6,7,8]
    
    for i in range(4):
    
        for j in simulations:
            
            modelName = 'Model-axon-%d' %i
            JobName = '4'
            UMAT = "umat_%d.f" %j
            timestep = 1.

            B = 3.

            # dimensions
            HH    = 1.
            LL    = 3.
            h       = 0.05 
            perturb = 0.0
            t     = h/4.
            waves = 1.  
            ptype = 2     # 0 = random, 1 = wavelength, 2 = center only, 3 = growth, 4 = mult. waves
        
            dimensions = [HH,LL,h,t,waves,perturb]

            # materials
            CorticalMaterial = "Cortex"
            SubCortMaterial = "Subcortex"

            # subcortical properties
            E         = 1.
            nu        = 0.45
            lam       = E*nu/((1+nu)*(1-2*nu))
            mu        = E/(2*(1+nu))
            alpha     = 4.75
            perturbation = 0
            tf = 0
            cr_pos = 1.0
            cr_neg = 1.0
            SubCortProps = [lam, mu, alpha, cr_pos, cr_neg, LL, perturbation, tf]    
            
            # cortical properties
            E         = E*B
            nu        = 0.45
            lam       = E*nu/((1+nu)*(1-2*nu))
            mu        = E/(2*(1+nu))
            g0        = alpha*10.**(params[i]) #0.24 #/day                    
            CorticalProps = [lam, mu, g0, 1., 0., 0.]

            # meshing
            bias = 5
            ecortex = 5
            esubcort = 45 
            et = 1 
            ew = 325
            meshing = [bias, ecortex, esubcort, et, ew]

            # timing
            InitialStep = T[i]/100.
            TotalTime = T[i]
            MinStep = T[i]/1000.
            MaxStep = T[i]/50.
            timing = [InitialStep, TotalTime, MinStep, MaxStep]

            # t0 = time.time()
            rectangle(modelName, dimensions, CorticalMaterial, CorticalProps, SubCortMaterial, SubCortProps, meshing, timing, JobName, UMAT, ptype)
            # t1 = time.time()
            # comptime = (t1-t0)/60.

        # modelName = 'Model-axon-orthotropic-%d' %i
        # JobName = 'axons-%d-ortho' %i
        # UMAT = "/home/mholla/Research/7 Scripts/umat_10.f"
        # dimensions = [HH,LL,h,t,waves,perturb]
        # meshing = [bias, ecortex, esubcort, et, ew] 
        # timing = [InitialStep, TotalTime, MinStep, MaxStep]
        # CorticalProps = [lam, mu, g0, 1., 0., 0.]
        # SubCortProps = [lam, mu, alpha, cr_pos, cr_pos, cr_pos, cr_neg, cr_neg, cr_neg, 1, 0, 0, 0, 1, 0, LL, perturbation, tf]  
        # rectangle(modelName, dimensions, CorticalMaterial, CorticalProps, SubCortMaterial, SubCortProps, meshing, timing, JobName, UMAT, ptype)
