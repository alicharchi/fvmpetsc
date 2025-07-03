static char help[] = "Solves 2D heat equation using fvm method.\n\n";

#include <petscksp.h>

void ijth(const PetscInt p,const PetscInt nx, PetscInt* i,PetscInt* j);
PetscInt pth(const PetscInt nx, const PetscInt i, const PetscInt j);

int main(int argc, char **args)
{
   Vec         x, b;                                      /* RHS */
   Mat         A;                                         /* linear system matrix */
   KSP         ksp;                                       /* linear solver context */
   PC          pc;                                        /* preconditioner context */
   PetscReal   tol = 1e-7; /* norm of solution error */
   PetscInt    i,j,p,k, nx = 100, ny = 50, N, col[5], rstart, rend, nlocal;
   PetscScalar value[5], bi;
   PetscScalar Lx=1,Ly=1;
   PetscScalar bcW =0, bcE = 0, bcN=0, bcS=0;
   PetscScalar aP, aW, aE, aS, aN, Sp, Su;
   PetscScalar alpha=1,dx, dy;
   PetscScalar Ae , Aw , As , An ;

   PetscFunctionBeginUser;
   PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-nx", &nx, NULL));
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-ny", &ny, NULL));
   PetscCall(PetscOptionsGetScalar(NULL, NULL, "-Lx", &Lx, NULL));
   PetscCall(PetscOptionsGetScalar(NULL, NULL, "-Ly", &Ly, NULL));
   PetscCall(PetscOptionsGetScalar(NULL, NULL, "-alpha", &alpha, NULL));
   PetscCall(PetscOptionsGetScalar(NULL, NULL, "-bcW", &bcW, NULL));
   PetscCall(PetscOptionsGetScalar(NULL, NULL, "-bcE", &bcE, NULL));
   PetscCall(PetscOptionsGetScalar(NULL, NULL, "-bcN", &bcN, NULL));
   PetscCall(PetscOptionsGetScalar(NULL, NULL, "-bcS", &bcS, NULL));

   N  = nx*ny;
   dx = Lx/nx;
   dy = Ly/ny;
   Ae = dy;
   Aw = dy;
   As = dx;
   An = dx;

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Compute the matrix and right-hand-side vector that define
   the linear system, Ax = b.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

   /*
   Create vectors.  Note that we form 1 vector from scratch and
   then duplicate as needed. For this simple case let PETSc decide how
   many elements of the vector are stored on each processor. The second
   argument to VecSetSizes() below causes PETSc to decide.
   */
   PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
   PetscCall(VecSetSizes(x, PETSC_DECIDE, N));
   PetscCall(VecSetFromOptions(x));
   PetscCall(VecDuplicate(x, &b));

   /* Identify the starting and ending mesh points on each
   processor for the interior part of the mesh. We let PETSc decide
   above. */

   PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
   PetscCall(VecGetLocalSize(x, &nlocal));

   /*
   Create matrix.  When using MatCreate(), the matrix format can
   be specified at runtime.

   Performance tuning note:  For problems of substantial size,
   preallocation of matrix memory is crucial for attaining good
   performance. See the matrix chapter of the users manual for details.

   We pass in nlocal as the "local" size of the matrix to force it
   to have the same parallel layout as the vector created above.
   */

   PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
   PetscCall(MatSetSizes(A, nlocal, nlocal, N, N));
   PetscCall(MatSetFromOptions(A));
   PetscCall(MatSetUp(A));

   /*
   Assemble matrix.

   The linear system is distributed across the processors by
   chunks of contiguous rows, which correspond to contiguous
   sections of the mesh on which the problem is discretized.
   For matrix assembly, each processor contributes entries for
   the part that it owns locally.
   */

   for (p = rstart; p < rend; p++) 
   {
      ijth(p,nx, &i,&j);
      k = 0; Sp = 0; Su = 0;
      if (i==0)
      {
         aW = 0;
         aE = alpha*Ae/dx;
         Sp += -2*alpha*Aw/dx;
         Su += (2*alpha*Aw/dx)*bcE;
      }
      else if (i==nx-1)
      {
         aW = alpha*Aw/dx;
         aE = 0;
         Sp += -2*alpha*Ae/dx;
         Su += (2*alpha*Ae/dx)*bcW;
      }
      else
      {
         aW = alpha*Aw/dx;
         aE = alpha*Ae/dx;
      }

      if (j==0){
         aS = 0;
         aN = alpha*An/dy;
         Sp += -2*alpha*As/dy;
         Su += (2*alpha*As/dy)*bcS;
      }
      else if (j==ny-1)
      {
         aS = alpha*As/dy;
         aN = 0;
         Sp += -2*alpha*An/dy;
         Su += (2*alpha*An/dy)*bcN;
      }
      else
      {
         aS = alpha*As/dy;
         aN = alpha*An/dy;
      }

      aP = aW + aE + aS + aN - Sp;
      col[k] = p; value[k] = aP; ++k;     

      if (j>0)
      {
         col[k] = pth(nx,i,j-1); value[k] = -aS; ++k;
      }
      if (j<ny-1)
      {
         col[k] = pth(nx,i,j+1); value[k] = -aN; ++k;
      }
      if (i>0)
      {
         col[k] = pth(nx,i-1,j); value[k] = -aW; ++k;
      }
      if (i<nx-1)
      {
         col[k] = pth(nx,i+1,j); value[k] = -aE; ++k;
      }

      bi = Su;

      PetscCall(MatSetValues(A, 1, &p, k, col, value, INSERT_VALUES));
      PetscCall(VecSetValues(b, 1, &p, &bi, INSERT_VALUES));
   }

   /* Assemble the matrix */
   PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

   PetscCall(VecAssemblyBegin(b));
   PetscCall(VecAssemblyEnd(b));
   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the linear solver and set various options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   /*
   Create linear solver context
   */
   PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));

   /*
   Set operators. Here the matrix that defines the linear system
   also serves as the preconditioning matrix.
   */
   PetscCall(KSPSetOperators(ksp, A, A));

   /*
   Set linear solver defaults for this problem (optional).
   - By extracting the KSP and PC contexts from the KSP context,
   we can then directly call any KSP and PC routines to set
   various options.
   - The following four statements are optional; all of these
   parameters could alternatively be specified at runtime via
   KSPSetFromOptions();
   */
   PetscCall(KSPGetPC(ksp, &pc));
   PetscCall(PCSetType(pc, PCJACOBI));
   PetscCall(KSPSetTolerances(ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));

   /*
   Set runtime options, e.g.,
   -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
   These options will override those specified above as long as
   KSPSetFromOptions() is called _after_ any other customization
   routines.
   */
   PetscCall(KSPSetFromOptions(ksp));

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Solve the linear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   /*
   Solve linear system
   */
   PetscCall(KSPSolve(ksp, b, x));

   /*
   View solver info; we could instead use the option -ksp_view to
   print this info to the screen at the conclusion of KSPSolve().
   */
   PetscCall(KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD));

   //Save temperature
   PetscViewer viewer;
   PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "T.out", &viewer));
   PetscCall(PetscViewerSetType(viewer ,PETSCVIEWERASCII));
   PetscCall(VecView(x, viewer));
   PetscCall(PetscViewerDestroy(&viewer));

   /*
   Free work space.  All PETSc objects should be destroyed when they
   are no longer needed.
   */
   PetscCall(VecDestroy(&x));
   PetscCall(VecDestroy(&b));
   PetscCall(MatDestroy(&A));
   PetscCall(KSPDestroy(&ksp));

   /*
   Always call PetscFinalize() before exiting a program.  This routine
   - finalizes the PETSc libraries as well as MPI
   - provides summary and diagnostic information if certain runtime
   options are chosen (e.g., -log_view).
   */
   PetscCall(PetscFinalize());
   return 0;
}

void ijth(const PetscInt p,const PetscInt nx, PetscInt* i,PetscInt* j)
{
   *j = p/nx;
   *i = p - (*j)*nx;    
}

PetscInt pth(const PetscInt nx, const PetscInt i, const PetscInt j)
{
   return j*nx + i;
}