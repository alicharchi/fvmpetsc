static char help[] = "Solves 1D heat equation using fvm method.\n\n";

#include <petscksp.h>

int main(int argc, char **args)
{
   Vec         x, b;                                      
   Mat         A;                                         
   KSP         ksp;                                       
   PC          pc;   
   PetscInt    i, k, n = 100, col[3], rstart, rend, nlocal;
   PetscScalar value[3], bi;
   PetscScalar L=1;
   PetscScalar bcW =0, bcE = 0;
   PetscScalar aP, aW, aE, Sp, Su;
   PetscScalar alpha=1, dx;
   PetscScalar Ae=1 , Aw=1;

   PetscFunctionBeginUser;
   PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
   PetscCall(PetscOptionsGetInt(NULL, NULL, "-nx", &n, NULL));
   PetscCall(PetscOptionsGetScalar(NULL, NULL, "-L", &L, NULL));   
   PetscCall(PetscOptionsGetScalar(NULL, NULL, "-alpha", &alpha, NULL));
   PetscCall(PetscOptionsGetScalar(NULL, NULL, "-bcW", &bcW, NULL));
   PetscCall(PetscOptionsGetScalar(NULL, NULL, "-bcE", &bcE, NULL));
   
   dx = L/n;   
   
   /* Create vectors */
   PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
   PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
   PetscCall(VecSetFromOptions(x));
   PetscCall(VecDuplicate(x, &b));

   /* Identify the starting and ending mesh points on each processor  */
   PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
   PetscCall(VecGetLocalSize(x, &nlocal));

   /* Create matrix */
   PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
   PetscCall(MatSetSizes(A, nlocal, nlocal, n, n));
   PetscCall(MatSetFromOptions(A));
   PetscCall(MatSetUp(A));

   /* Assemble matrix */
   for (i = rstart; i < rend; i++) 
   {      
      k = 0; Sp = 0; Su = 0;
      if (i==0)
      {
         aW = 0;
         aE = alpha*Ae/dx;
         Sp += -2*alpha*Aw/dx;
         Su += (2*alpha*Aw/dx)*bcE;
      }
      else if (i==n-1)
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

      aP = aW + aE - Sp;
      col[k] = i; value[k] = aP; ++k;     

      if (i>0)
      {
         col[k] = i-1; value[k] = -aW; ++k;
      }
      if (i<n-1)
      {
         col[k] = i+1; value[k] = -aE; ++k;
      }

      bi = Su;

      PetscCall(MatSetValues(A, 1, &i, k, col, value, INSERT_VALUES));
      PetscCall(VecSetValues(b, 1, &i, &bi, INSERT_VALUES));
   }

   /* Assemble the matrix */
   PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
   PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

   PetscCall(VecAssemblyBegin(b));
   PetscCall(VecAssemblyEnd(b));

   /* Create linear solver context */
   PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));

   PetscCall(KSPSetOperators(ksp, A, A));   
   PetscCall(KSPGetPC(ksp, &pc));
   PetscCall(PCSetType(pc, PCJACOBI));   
   PetscCall(KSPSetFromOptions(ksp));

   /* Solve linear system */
   PetscCall(KSPSolve(ksp, b, x));
   
   PetscCall(KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD));

   /* Save temperature */
   PetscViewer viewer;
   PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "T.out", &viewer));
   PetscCall(PetscViewerSetType(viewer ,PETSCVIEWERASCII));
   PetscCall(VecView(x, viewer));
   PetscCall(PetscViewerDestroy(&viewer));

   /* Free work space */
   PetscCall(VecDestroy(&x));
   PetscCall(VecDestroy(&b));
   PetscCall(MatDestroy(&A));
   PetscCall(KSPDestroy(&ksp));
   
   PetscCall(PetscFinalize());
   return 0;
}