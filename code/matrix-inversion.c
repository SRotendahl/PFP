/*
 * Headers
*/

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


/*
 * Initialisation
*/

int futhark_get_num_sizes(void);
const char *futhark_get_size_name(int);
const char *futhark_get_size_class(int);
struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_add_build_option(struct futhark_context_config *cfg,
                                             const char *opt);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_profiling(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s);
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s);
void
futhark_context_config_select_device_interactively(struct futhark_context_config *cfg);
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path);
void
futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                         const char *path);
void futhark_context_config_dump_binary_to(struct futhark_context_config *cfg,
                                           const char *path);
void futhark_context_config_load_binary_from(struct futhark_context_config *cfg,
                                             const char *path);
void
futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                              int size);
void
futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                              int num);
void
futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                             int num);
void
futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                             int num);
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
struct futhark_context
*futhark_context_new_with_command_queue(struct futhark_context_config *cfg,
                                        cl_command_queue queue);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
cl_command_queue futhark_context_get_command_queue(struct futhark_context *ctx);

/*
 * Arrays
*/

struct futhark_f32_3d ;
struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx,
                                          float *data, int64_t dim0,
                                          int64_t dim1, int64_t dim2);
struct futhark_f32_3d *futhark_new_raw_f32_3d(struct futhark_context *ctx,
                                              cl_mem data, int offset,
                                              int64_t dim0, int64_t dim1,
                                              int64_t dim2);
int futhark_free_f32_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d *arr);
int futhark_values_f32_3d(struct futhark_context *ctx,
                          struct futhark_f32_3d *arr, float *data);
cl_mem futhark_values_raw_f32_3d(struct futhark_context *ctx,
                                 struct futhark_f32_3d *arr);
int64_t *futhark_shape_f32_3d(struct futhark_context *ctx,
                              struct futhark_f32_3d *arr);

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_3d **out0, const
                       struct futhark_f32_3d *in0);

/*
 * Miscellaneous
*/

void futhark_debugging_report(struct futhark_context *ctx);
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
// Start of panic.h.

#include <stdarg.h>

static const char *fut_progname;

static void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
        fprintf(stderr, "%s: ", fut_progname);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* For generating arbitrary-sized error messages.  It is the callers
   responsibility to free the buffer at some point. */
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*) malloc(needed);
  va_start(vl, s); /* Must re-init. */
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}

// End of panic.h.

// Start of timing.h.

// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

// End of timing.h.

#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
// Start of values.h.

//// Text I/O

typedef int (*writer)(FILE*, void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces() {
  int c;
  do {
    c = getchar();
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(char *buf, int bufsize) {
 start:
  skipspaces();

  int i = 0;
  while (i < bufsize) {
    int c = getchar();
    buf[i] = c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getchar());
      goto start;
    } else if (!constituent(c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, stdin);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(char *buf, int bufsize, const char* expected) {
  next_token(buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    reader->n_elems_space * reader->elem_size);
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(char *buf, int bufsize,
                                struct array_reader *reader, int dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc(dims,sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc(dims,sizeof(int64_t));

  while (1) {
    next_token(buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  shape[0] = 0;

  if (!next_token_is(buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 1; i < dims; i++) {
    if (!next_token_is(buf, bufsize, "[")) {
      return 1;
    }

    next_token(buf, bufsize);

    if (sscanf(buf, "%d", &shape[i]) != 1) {
      return 1;
    }

    if (!next_token_is(buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(buf, bufsize, ")")) {
    return 1;
  }

  return 0;
}

static int read_str_array(int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, elem_size*reader.n_elems_space);
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNi8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNu8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

static void flip_bytes(int elem_size, unsigned char *elem) {
  for (int j=0; j<elem_size/2; j++) {
    unsigned char head = elem[j];
    int tail_index = elem_size-1-j;
    elem[j] = elem[tail_index];
    elem[tail_index] = head;
  }
}

// On Windows we need to explicitly set the file mode to not mangle
// newline characters.  On *nix there is no difference.
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static void set_binary_mode(FILE *f) {
  setmode(fileno(f), O_BINARY);
}
#else
static void set_binary_mode(FILE *f) {
  (void)f;
}
#endif

static int read_byte(void* dest) {
  int num_elems_read = fread(dest, 1, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary() {
  skipspaces();
  int c = getchar();
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(&bin_version);

    if (ret != 0) { panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, stdin);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum() {
  char read_binname[4];

  int num_matched = scanf("%4c", read_binname);
  if (num_matched != 1) { panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum();
  if (bin_type != expected_type) {
    panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum();
  if (expected_type != bin_primtype) {
    panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  uint64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    uint64_t bin_shape;
    ret = fread(&bin_shape, sizeof(bin_shape), 1, stdin);
    if (ret != 1) {
      panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i);
    }
    if (IS_BIG_ENDIAN) {
      flip_bytes(sizeof(bin_shape), (unsigned char*) &bin_shape);
    }
    elem_count *= bin_shape;
    shape[i] = (int64_t) bin_shape;
  }

  size_t elem_size = expected_type->size;
  void* tmp = realloc(*data, elem_count * elem_size);
  if (tmp == NULL) {
    panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  size_t num_elems_read = fread(*data, elem_size, elem_count, stdin);
  if (num_elems_read != elem_count) {
    panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    flip_bytes(elem_size, (unsigned char*) *data);
  }

  return 0;
}

static int read_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary()) {
    return read_str_array(expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(expected_type, data, shape, dims);
  }
}

static int write_str_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int64_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      printf("empty(");
      for (int64_t i = 1; i < rank; i++) {
        printf("[%d]", shape[i]);
      }
      printf("%s", elem_type->type_name);
      printf(")");
    } else if (rank==1) {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    } else {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    }
  }
  return 0;
}

static int write_bin_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fputs(elem_type->binname, out);
  if (shape != NULL) {
    fwrite(shape, sizeof(int64_t), rank, out);
  }

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, elem_type->size, num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type, void *data, int64_t *shape, int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary()) {
    char buf[100];
    next_token(buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(expected_type);
    size_t elem_size = expected_type->size;
    int num_elems_read = fread(dest, elem_size, 1, stdin);
    if (IS_BIG_ENDIAN) {
      flip_bytes(elem_size, (unsigned char*) dest);
    }
    return num_elems_read == 1 ? 0 : 1;
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

// End of values.h.

static int binary_output = 0;
static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
// Start of tuning.h.

static char* load_tuning_file(const char *fname,
                              void *cfg,
                              int (*set_size)(void*, const char*, size_t)) {
  const int max_line_len = 1024;
  char* line = (char*) malloc(max_line_len);

  FILE *f = fopen(fname, "r");

  if (f == NULL) {
    snprintf(line, max_line_len, "Cannot open file: %s", strerror(errno));
    return line;
  }

  int lineno = 0;
  while (fgets(line, max_line_len, f) != NULL) {
    lineno++;
    char *eql = strstr(line, "=");
    if (eql) {
      *eql = 0;
      int value = atoi(eql+1);
      if (set_size(cfg, line, value) != 0) {
        strncpy(eql+1, line, max_line_len-strlen(line)-1);
        snprintf(line, max_line_len, "Unknown name '%s' on line %d.", eql+1, lineno);
        return line;
      }
    } else {
      snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
               lineno);
      return line;
    }
  }

  free(line);

  return NULL;
}

// End of tuning.h.

int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"debugging", no_argument, NULL, 3},
                                           {"log", no_argument, NULL, 4},
                                           {"entry-point", required_argument,
                                            NULL, 5}, {"binary-output",
                                                       no_argument, NULL, 6},
                                           {"platform", required_argument, NULL,
                                            7}, {"device", required_argument,
                                                 NULL, 8},
                                           {"default-group-size",
                                            required_argument, NULL, 9},
                                           {"default-num-groups",
                                            required_argument, NULL, 10},
                                           {"default-tile-size",
                                            required_argument, NULL, 11},
                                           {"default-threshold",
                                            required_argument, NULL, 12},
                                           {"dump-opencl", required_argument,
                                            NULL, 13}, {"load-opencl",
                                                        required_argument, NULL,
                                                        14},
                                           {"dump-opencl-binary",
                                            required_argument, NULL, 15},
                                           {"load-opencl-binary",
                                            required_argument, NULL, 16},
                                           {"build-option", required_argument,
                                            NULL, 17}, {"print-sizes",
                                                        no_argument, NULL, 18},
                                           {"size", required_argument, NULL,
                                            19}, {"tuning", required_argument,
                                                  NULL, 20}, {"profile",
                                                              no_argument, NULL,
                                                              21}, {0, 0, 0,
                                                                    0}};
    
    while ((ch = getopt_long(argc, argv, ":t:r:DLe:bp:d:P", long_options,
                             NULL)) != -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                panic(1, "Cannot open %s: %s\n", optarg, strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                panic(1, "Need a positive number of runs, not %s\n", optarg);
        }
        if (ch == 3 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 4 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 5 || ch == 'e') {
            if (entry_point != NULL)
                entry_point = optarg;
        }
        if (ch == 6 || ch == 'b')
            binary_output = 1;
        if (ch == 7 || ch == 'p')
            futhark_context_config_set_platform(cfg, optarg);
        if (ch == 8 || ch == 'd')
            futhark_context_config_set_device(cfg, optarg);
        if (ch == 9)
            futhark_context_config_set_default_group_size(cfg, atoi(optarg));
        if (ch == 10)
            futhark_context_config_set_default_num_groups(cfg, atoi(optarg));
        if (ch == 11)
            futhark_context_config_set_default_tile_size(cfg, atoi(optarg));
        if (ch == 12)
            futhark_context_config_set_default_threshold(cfg, atoi(optarg));
        if (ch == 13) {
            futhark_context_config_dump_program_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 14)
            futhark_context_config_load_program_from(cfg, optarg);
        if (ch == 15) {
            futhark_context_config_dump_binary_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 16)
            futhark_context_config_load_binary_from(cfg, optarg);
        if (ch == 17)
            futhark_context_config_add_build_option(cfg, optarg);
        if (ch == 18) {
            int n = futhark_get_num_sizes();
            
            for (int i = 0; i < n; i++)
                printf("%s (%s)\n", futhark_get_size_name(i),
                       futhark_get_size_class(i));
            exit(0);
        }
        if (ch == 19) {
            char *name = optarg;
            char *equals = strstr(optarg, "=");
            char *value_str = equals != NULL ? equals + 1 : optarg;
            int value = atoi(value_str);
            
            if (equals != NULL) {
                *equals = 0;
                if (futhark_context_config_set_size(cfg, name, value) != 0)
                    panic(1, "Unknown size: %s\n", name);
            } else
                panic(1, "Invalid argument for size option: %s\n", optarg);
        }
        if (ch == 20) {
            char *ret = load_tuning_file(optarg, cfg, (int (*)(void *, const
                                                               char *,
                                                               size_t)) futhark_context_config_set_size);
            
            if (ret != NULL)
                panic(1, "When loading tuning from '%s': %s\n", optarg, ret);
        }
        if (ch == 21 || ch == 'P')
            futhark_context_config_set_profiling(cfg, 1);
        if (ch == ':')
            panic(-1, "Missing argument for option %s\n", argv[optind - 1]);
        if (ch == '?') {
            fprintf(stderr, "Usage: %s: %s\n", fut_progname,
                    "[-t/--write-runtime-to FILE] [-r/--runs INT] [-D/--debugging] [-L/--log] [-e/--entry-point NAME] [-b/--binary-output] [-p/--platform NAME] [-d/--device NAME] [--default-group-size INT] [--default-num-groups INT] [--default-tile-size INT] [--default-threshold INT] [--dump-opencl FILE] [--load-opencl FILE] [--dump-opencl-binary FILE] [--load-opencl-binary FILE] [--build-option OPT] [--print-sizes] [--size NAME=INT] [--tuning FILE] [-P/--profile]");
            panic(1, "Unknown option: %s\n", argv[optind - 1]);
        }
    }
    return optind;
}
static void futrts_cli_entry_main(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    struct futhark_f32_3d *read_value_10247;
    int64_t read_shape_10248[3];
    float *read_arr_10249 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_10249, read_shape_10248, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *result_10250;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_10247 = futhark_new_f32_3d(ctx, read_arr_10249,
                                                      read_shape_10248[0],
                                                      read_shape_10248[1],
                                                      read_shape_10248[2])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_10250, read_value_10247);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_10247) == 0);
        assert(futhark_free_f32_3d(ctx, result_10250) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_10247 = futhark_new_f32_3d(ctx, read_arr_10249,
                                                      read_shape_10248[0],
                                                      read_shape_10248[1],
                                                      read_shape_10248[2])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_10250, read_value_10247);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_10247) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_3d(ctx, result_10250) == 0);
        }
    }
    free(read_arr_10249);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_10250)[0] *
                            futhark_shape_f32_3d(ctx, result_10250)[1] *
                            futhark_shape_f32_3d(ctx, result_10250)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_10250, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_10250), 3);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_3d(ctx, result_10250) == 0);
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="main", .fun =
                                                futrts_cli_entry_main}};
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    if (entry_point != NULL) {
        int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
        entry_point_fun *entry_point_fun = NULL;
        
        for (int i = 0; i < num_entry_points; i++) {
            if (strcmp(entry_points[i].name, entry_point) == 0) {
                entry_point_fun = entry_points[i].fun;
                break;
            }
        }
        if (entry_point_fun == NULL) {
            fprintf(stderr,
                    "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                    entry_point);
            for (int i = 0; i < num_entry_points; i++)
                fprintf(stderr, "%s\n", entry_points[i].name);
            return 1;
        }
        entry_point_fun(ctx);
        if (runtime_file != NULL)
            fclose(runtime_file);
        futhark_debugging_report(ctx);
    }
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
// Start of lock.h.

/* A very simple cross-platform implementation of locks.  Uses
   pthreads on Unix and some Windows thing there.  Futhark's
   host-level code is not multithreaded, but user code may be, so we
   need some mechanism for ensuring atomic access to API functions.
   This is that mechanism.  It is not exposed to user code at all, so
   we do not have to worry about name collisions. */

#ifdef _WIN32

typedef HANDLE lock_t;

static lock_t create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  /* Default security attributes. */
                      FALSE, /* Initially unlocked. */
                      NULL); /* Unnamed. */
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
/* Assuming POSIX */

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  /* Nothing to do for pthreads. */
  (void)lock;
}

#endif

// End of lock.h.

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_SILENCE_DEPRECATION // For macOS.
#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif
typedef cl_mem fl_mem_t;
// Start of free_list.h.

/* An entry in the free list.  May be invalid, to avoid having to
   deallocate entries as soon as they are removed.  There is also a
   tag, to help with memory reuse. */
struct free_list_entry {
  size_t size;
  fl_mem_t mem;
  const char *tag;
  unsigned char valid;
};

struct free_list {
  struct free_list_entry *entries;        // Pointer to entries.
  int capacity;                           // Number of entries.
  int used;                               // Number of valid entries.
};

static void free_list_init(struct free_list *l) {
  l->capacity = 30; // Picked arbitrarily.
  l->used = 0;
  l->entries = (struct free_list_entry*) malloc(sizeof(struct free_list_entry) * l->capacity);
  for (int i = 0; i < l->capacity; i++) {
    l->entries[i].valid = 0;
  }
}

/* Remove invalid entries from the free list. */
static void free_list_pack(struct free_list *l) {
  int p = 0;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[p] = l->entries[i];
      p++;
    }
  }
  // Now p == l->used.
  l->entries = realloc(l->entries, l->used * sizeof(struct free_list_entry));
  l->capacity = l->used;
}

static void free_list_destroy(struct free_list *l) {
  assert(l->used == 0);
  free(l->entries);
}

static int free_list_find_invalid(struct free_list *l) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (!l->entries[i].valid) {
      break;
    }
  }
  return i;
}

static void free_list_insert(struct free_list *l, size_t size, fl_mem_t mem, const char *tag) {
  int i = free_list_find_invalid(l);

  if (i == l->capacity) {
    // List is full; so we have to grow it.
    int new_capacity = l->capacity * 2 * sizeof(struct free_list_entry);
    l->entries = realloc(l->entries, new_capacity);
    for (int j = 0; j < l->capacity; j++) {
      l->entries[j+l->capacity].valid = 0;
    }
    l->capacity *= 2;
  }

  // Now 'i' points to the first invalid entry.
  l->entries[i].valid = 1;
  l->entries[i].size = size;
  l->entries[i].mem = mem;
  l->entries[i].tag = tag;

  l->used++;
}

/* Find and remove a memory block of at least the desired size and
   tag.  Returns 0 on success.  */
static int free_list_find(struct free_list *l, const char *tag, size_t *size_out, fl_mem_t *mem_out) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid && l->entries[i].tag == tag) {
      l->entries[i].valid = 0;
      *size_out = l->entries[i].size;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

/* Remove the first block in the free list.  Returns 0 if a block was
   removed, and nonzero if the free list was already empty. */
static int free_list_first(struct free_list *l, fl_mem_t *mem_out) {
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[i].valid = 0;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

// End of free_list.h.

// Start of opencl.h.

#define OPENCL_SUCCEED_FATAL(e) opencl_succeed_fatal(e, #e, __FILE__, __LINE__)
#define OPENCL_SUCCEED_NONFATAL(e) opencl_succeed_nonfatal(e, #e, __FILE__, __LINE__)
// Take care not to override an existing error.
#define OPENCL_SUCCEED_OR_RETURN(e) {             \
    char *error = OPENCL_SUCCEED_NONFATAL(e);     \
    if (error) {                                  \
      if (!ctx->error) {                          \
        ctx->error = error;                       \
        return bad;                               \
      } else {                                    \
        free(error);                              \
      }                                           \
    }                                             \
  }

// OPENCL_SUCCEED_OR_RETURN returns the value of the variable 'bad' in
// scope.  By default, it will be this one.  Create a local variable
// of some other type if needed.  This is a bit of a hack, but it
// saves effort in the code generator.
static const int bad = 1;

struct opencl_config {
  int debugging;
  int profiling;
  int logging;
  int preferred_device_num;
  const char *preferred_platform;
  const char *preferred_device;
  int ignore_blacklist;

  const char* dump_program_to;
  const char* load_program_from;
  const char* dump_binary_to;
  const char* load_binary_from;

  size_t default_group_size;
  size_t default_num_groups;
  size_t default_tile_size;
  size_t default_threshold;

  int default_group_size_changed;
  int default_tile_size_changed;

  int num_sizes;
  const char **size_names;
  const char **size_vars;
  size_t *size_values;
  const char **size_classes;
};

static void opencl_config_init(struct opencl_config *cfg,
                               int num_sizes,
                               const char *size_names[],
                               const char *size_vars[],
                               size_t *size_values,
                               const char *size_classes[]) {
  cfg->debugging = 0;
  cfg->logging = 0;
  cfg->profiling = 0;
  cfg->preferred_device_num = 0;
  cfg->preferred_platform = "";
  cfg->preferred_device = "";
  cfg->ignore_blacklist = 0;
  cfg->dump_program_to = NULL;
  cfg->load_program_from = NULL;
  cfg->dump_binary_to = NULL;
  cfg->load_binary_from = NULL;

  // The following are dummy sizes that mean the concrete defaults
  // will be set during initialisation via hardware-inspection-based
  // heuristics.
  cfg->default_group_size = 0;
  cfg->default_num_groups = 0;
  cfg->default_tile_size = 0;
  cfg->default_threshold = 0;

  cfg->default_group_size_changed = 0;
  cfg->default_tile_size_changed = 0;

  cfg->num_sizes = num_sizes;
  cfg->size_names = size_names;
  cfg->size_vars = size_vars;
  cfg->size_values = size_values;
  cfg->size_classes = size_classes;
}

// A record of something that happened.
struct profiling_record {
  cl_event *event;
  int *runs;
  int64_t *runtime;
};

struct opencl_context {
  cl_device_id device;
  cl_context ctx;
  cl_command_queue queue;

  struct opencl_config cfg;

  struct free_list free_list;

  size_t max_group_size;
  size_t max_num_groups;
  size_t max_tile_size;
  size_t max_threshold;
  size_t max_local_memory;

  size_t lockstep_width;

  struct profiling_record *profiling_records;
  int profiling_records_capacity;
  int profiling_records_used;
};

struct opencl_device_option {
  cl_platform_id platform;
  cl_device_id device;
  cl_device_type device_type;
  char *platform_name;
  char *device_name;
};

/* This function must be defined by the user.  It is invoked by
   setup_opencl() after the platform and device has been found, but
   before the program is loaded.  Its intended use is to tune
   constants based on the selected platform and device. */
static void post_opencl_setup(struct opencl_context*, struct opencl_device_option*);

static char *strclone(const char *str) {
  size_t size = strlen(str) + 1;
  char *copy = (char*) malloc(size);
  if (copy == NULL) {
    return NULL;
  }

  memcpy(copy, str, size);
  return copy;
}

// Read a file into a NUL-terminated string; returns NULL on error.
static char* slurp_file(const char *filename, size_t *size) {
  char *s;
  FILE *f = fopen(filename, "rb"); // To avoid Windows messing with linebreaks.
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  size_t src_size = ftell(f);
  fseek(f, 0, SEEK_SET);
  s = (char*) malloc(src_size + 1);
  if (fread(s, 1, src_size, f) != src_size) {
    free(s);
    s = NULL;
  } else {
    s[src_size] = '\0';
  }
  fclose(f);

  if (size) {
    *size = src_size;
  }

  return s;
}

static const char* opencl_error_string(cl_int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown";
    }
}

static void opencl_succeed_fatal(unsigned int ret,
                                 const char *call,
                                 const char *file,
                                 int line) {
  if (ret != CL_SUCCESS) {
    panic(-1, "%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
          file, line, call, ret, opencl_error_string(ret));
  }
}

static char* opencl_succeed_nonfatal(unsigned int ret,
                                     const char *call,
                                     const char *file,
                                     int line) {
  if (ret != CL_SUCCESS) {
    return msgprintf("%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
                     file, line, call, ret, opencl_error_string(ret));
  } else {
    return NULL;
  }
}

static void set_preferred_platform(struct opencl_config *cfg, const char *s) {
  cfg->preferred_platform = s;
  cfg->ignore_blacklist = 1;
}

static void set_preferred_device(struct opencl_config *cfg, const char *s) {
  int x = 0;
  if (*s == '#') {
    s++;
    while (isdigit(*s)) {
      x = x * 10 + (*s++)-'0';
    }
    // Skip trailing spaces.
    while (isspace(*s)) {
      s++;
    }
  }
  cfg->preferred_device = s;
  cfg->preferred_device_num = x;
  cfg->ignore_blacklist = 1;
}

static char* opencl_platform_info(cl_platform_id platform,
                                  cl_platform_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED_FATAL(clGetPlatformInfo(platform, param, 0, NULL, &req_bytes));

  info = (char*) malloc(req_bytes);

  OPENCL_SUCCEED_FATAL(clGetPlatformInfo(platform, param, req_bytes, info, NULL));

  return info;
}

static char* opencl_device_info(cl_device_id device,
                                cl_device_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device, param, 0, NULL, &req_bytes));

  info = (char*) malloc(req_bytes);

  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device, param, req_bytes, info, NULL));

  return info;
}

static void opencl_all_device_options(struct opencl_device_option **devices_out,
                                      size_t *num_devices_out) {
  size_t num_devices = 0, num_devices_added = 0;

  cl_platform_id *all_platforms;
  cl_uint *platform_num_devices;

  cl_uint num_platforms;

  // Find the number of platforms.
  OPENCL_SUCCEED_FATAL(clGetPlatformIDs(0, NULL, &num_platforms));

  // Make room for them.
  all_platforms = calloc(num_platforms, sizeof(cl_platform_id));
  platform_num_devices = calloc(num_platforms, sizeof(cl_uint));

  // Fetch all the platforms.
  OPENCL_SUCCEED_FATAL(clGetPlatformIDs(num_platforms, all_platforms, NULL));

  // Count the number of devices for each platform, as well as the
  // total number of devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    if (clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL,
                       0, NULL, &platform_num_devices[i]) == CL_SUCCESS) {
      num_devices += platform_num_devices[i];
    } else {
      platform_num_devices[i] = 0;
    }
  }

  // Make room for all the device options.
  struct opencl_device_option *devices =
    calloc(num_devices, sizeof(struct opencl_device_option));

  // Loop through the platforms, getting information about their devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    cl_platform_id platform = all_platforms[i];
    cl_uint num_platform_devices = platform_num_devices[i];

    if (num_platform_devices == 0) {
      continue;
    }

    char *platform_name = opencl_platform_info(platform, CL_PLATFORM_NAME);
    cl_device_id *platform_devices =
      calloc(num_platform_devices, sizeof(cl_device_id));

    // Fetch all the devices.
    OPENCL_SUCCEED_FATAL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                  num_platform_devices, platform_devices, NULL));

    // Loop through the devices, adding them to the devices array.
    for (cl_uint i = 0; i < num_platform_devices; i++) {
      char *device_name = opencl_device_info(platform_devices[i], CL_DEVICE_NAME);
      devices[num_devices_added].platform = platform;
      devices[num_devices_added].device = platform_devices[i];
      OPENCL_SUCCEED_FATAL(clGetDeviceInfo(platform_devices[i], CL_DEVICE_TYPE,
                                     sizeof(cl_device_type),
                                     &devices[num_devices_added].device_type,
                                     NULL));
      // We don't want the structs to share memory, so copy the platform name.
      // Each device name is already unique.
      devices[num_devices_added].platform_name = strclone(platform_name);
      devices[num_devices_added].device_name = device_name;
      num_devices_added++;
    }
    free(platform_devices);
    free(platform_name);
  }
  free(all_platforms);
  free(platform_num_devices);

  *devices_out = devices;
  *num_devices_out = num_devices;
}

// Returns 0 on success.
static int select_device_interactively(struct opencl_config *cfg) {
  struct opencl_device_option *devices;
  size_t num_devices;
  int ret = 1;

  opencl_all_device_options(&devices, &num_devices);

  printf("Choose OpenCL device:\n");
  const char *cur_platform = "";
  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (strcmp(cur_platform, device.platform_name) != 0) {
      printf("Platform: %s\n", device.platform_name);
      cur_platform = device.platform_name;
    }
    printf("[%d] %s\n", (int)i, device.device_name);
  }

  int selection;
  printf("Choice: ");
  if (scanf("%d", &selection) == 1) {
    ret = 0;
    cfg->preferred_platform = "";
    cfg->preferred_device = "";
    cfg->preferred_device_num = selection;
    cfg->ignore_blacklist = 1;
  }

  // Free all the platform and device names.
  for (size_t j = 0; j < num_devices; j++) {
    free(devices[j].platform_name);
    free(devices[j].device_name);
  }
  free(devices);

  return ret;
}

static int is_blacklisted(const char *platform_name, const char *device_name,
                          const struct opencl_config *cfg) {
  if (strcmp(cfg->preferred_platform, "") != 0 ||
      strcmp(cfg->preferred_device, "") != 0) {
    return 0;
  } else if (strstr(platform_name, "Apple") != NULL &&
             strstr(device_name, "Intel(R) Core(TM)") != NULL) {
    return 1;
  } else {
    return 0;
  }
}

static struct opencl_device_option get_preferred_device(const struct opencl_config *cfg) {
  struct opencl_device_option *devices;
  size_t num_devices;

  opencl_all_device_options(&devices, &num_devices);

  int num_device_matches = 0;

  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (strstr(device.platform_name, cfg->preferred_platform) != NULL &&
        strstr(device.device_name, cfg->preferred_device) != NULL &&
        (cfg->ignore_blacklist ||
         !is_blacklisted(device.platform_name, device.device_name, cfg)) &&
        num_device_matches++ == cfg->preferred_device_num) {
      // Free all the platform and device names, except the ones we have chosen.
      for (size_t j = 0; j < num_devices; j++) {
        if (j != i) {
          free(devices[j].platform_name);
          free(devices[j].device_name);
        }
      }
      free(devices);
      return device;
    }
  }

  panic(1, "Could not find acceptable OpenCL device.\n");
  exit(1); // Never reached
}

static void describe_device_option(struct opencl_device_option device) {
  fprintf(stderr, "Using platform: %s\n", device.platform_name);
  fprintf(stderr, "Using device: %s\n", device.device_name);
}

static cl_build_status build_opencl_program(cl_program program, cl_device_id device, const char* options) {
  cl_int clBuildProgram_error = clBuildProgram(program, 1, &device, options, NULL, NULL);

  // Avoid termination due to CL_BUILD_PROGRAM_FAILURE
  if (clBuildProgram_error != CL_SUCCESS &&
      clBuildProgram_error != CL_BUILD_PROGRAM_FAILURE) {
    OPENCL_SUCCEED_FATAL(clBuildProgram_error);
  }

  cl_build_status build_status;
  OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program,
                                             device,
                                             CL_PROGRAM_BUILD_STATUS,
                                             sizeof(cl_build_status),
                                             &build_status,
                                             NULL));

  if (build_status != CL_SUCCESS) {
    char *build_log;
    size_t ret_val_size;
    OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size));

    build_log = (char*) malloc(ret_val_size+1);
    OPENCL_SUCCEED_FATAL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL));

    // The spec technically does not say whether the build log is zero-terminated, so let's be careful.
    build_log[ret_val_size] = '\0';

    fprintf(stderr, "Build log:\n%s\n", build_log);

    free(build_log);
  }

  return build_status;
}

/* Fields in a bitmask indicating which types we must be sure are
   available. */
enum opencl_required_type { OPENCL_F64 = 1 };

// We take as input several strings representing the program, because
// C does not guarantee that the compiler supports particularly large
// literals.  Notably, Visual C has a limit of 2048 characters.  The
// array must be NULL-terminated.
static cl_program setup_opencl_with_command_queue(struct opencl_context *ctx,
                                                  cl_command_queue queue,
                                                  const char *srcs[],
                                                  int required_types,
                                                  const char *extra_build_opts[]) {
  int error;

  ctx->queue = queue;

  OPENCL_SUCCEED_FATAL(clGetCommandQueueInfo(ctx->queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx->ctx, NULL));

  // Fill out the device info.  This is redundant work if we are
  // called from setup_opencl() (which is the common case), but I
  // doubt it matters much.
  struct opencl_device_option device_option;
  OPENCL_SUCCEED_FATAL(clGetCommandQueueInfo(ctx->queue, CL_QUEUE_DEVICE,
                                       sizeof(cl_device_id),
                                       &device_option.device,
                                       NULL));
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_PLATFORM,
                                 sizeof(cl_platform_id),
                                 &device_option.platform,
                                 NULL));
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_TYPE,
                                 sizeof(cl_device_type),
                                 &device_option.device_type,
                                 NULL));
  device_option.platform_name = opencl_platform_info(device_option.platform, CL_PLATFORM_NAME);
  device_option.device_name = opencl_device_info(device_option.device, CL_DEVICE_NAME);

  ctx->device = device_option.device;

  if (required_types & OPENCL_F64) {
    cl_uint supported;
    OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                                   sizeof(cl_uint), &supported, NULL));
    if (!supported) {
      panic(1, "Program uses double-precision floats, but this is not supported on the chosen device: %s\n",
            device_option.device_name);
    }
  }

  size_t max_group_size;
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof(size_t), &max_group_size, NULL));

  size_t max_tile_size = sqrt(max_group_size);

  cl_ulong max_local_memory;
  OPENCL_SUCCEED_FATAL(clGetDeviceInfo(device_option.device, CL_DEVICE_LOCAL_MEM_SIZE,
                                       sizeof(size_t), &max_local_memory, NULL));

  // Make sure this function is defined.
  post_opencl_setup(ctx, &device_option);

  if (max_group_size < ctx->cfg.default_group_size) {
    if (ctx->cfg.default_group_size_changed) {
      fprintf(stderr, "Note: Device limits default group size to %zu (down from %zu).\n",
              max_group_size, ctx->cfg.default_group_size);
    }
    ctx->cfg.default_group_size = max_group_size;
  }

  if (max_tile_size < ctx->cfg.default_tile_size) {
    if (ctx->cfg.default_tile_size_changed) {
      fprintf(stderr, "Note: Device limits default tile size to %zu (down from %zu).\n",
              max_tile_size, ctx->cfg.default_tile_size);
    }
    ctx->cfg.default_tile_size = max_tile_size;
  }

  ctx->max_group_size = max_group_size;
  ctx->max_tile_size = max_tile_size; // No limit.
  ctx->max_threshold = ctx->max_num_groups = 0; // No limit.
  ctx->max_local_memory = max_local_memory;

  // Now we go through all the sizes, clamp them to the valid range,
  // or set them to the default.
  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    const char *size_class = ctx->cfg.size_classes[i];
    size_t *size_value = &ctx->cfg.size_values[i];
    const char* size_name = ctx->cfg.size_names[i];
    size_t max_value, default_value;
    if (strstr(size_class, "group_size") == size_class) {
      max_value = max_group_size;
      default_value = ctx->cfg.default_group_size;
    } else if (strstr(size_class, "num_groups") == size_class) {
      max_value = max_group_size; // Futhark assumes this constraint.
      default_value = ctx->cfg.default_num_groups;
    } else if (strstr(size_class, "tile_size") == size_class) {
      max_value = sqrt(max_group_size);
      default_value = ctx->cfg.default_tile_size;
    } else if (strstr(size_class, "threshold") == size_class) {
      max_value = 0; // No limit.
      default_value = ctx->cfg.default_threshold;
    } else {
      panic(1, "Unknown size class for size '%s': %s\n", size_name, size_class);
    }
    if (*size_value == 0) {
      *size_value = default_value;
    } else if (max_value > 0 && *size_value > max_value) {
      fprintf(stderr, "Note: Device limits %s to %d (down from %d)\n",
              size_name, (int)max_value, (int)*size_value);
      *size_value = max_value;
    }
  }

  if (ctx->lockstep_width == 0) {
    ctx->lockstep_width = 1;
  }

  if (ctx->cfg.logging) {
    fprintf(stderr, "Lockstep width: %d\n", (int)ctx->lockstep_width);
    fprintf(stderr, "Default group size: %d\n", (int)ctx->cfg.default_group_size);
    fprintf(stderr, "Default number of groups: %d\n", (int)ctx->cfg.default_num_groups);
  }

  char *fut_opencl_src = NULL;
  size_t src_size = 0;

  // Maybe we have to read OpenCL source from somewhere else (used for debugging).
  if (ctx->cfg.load_program_from != NULL) {
    fut_opencl_src = slurp_file(ctx->cfg.load_program_from, NULL);
    assert(fut_opencl_src != NULL);
  } else {
    // Build the OpenCL program.  First we have to concatenate all the fragments.
    for (const char **src = srcs; src && *src; src++) {
      src_size += strlen(*src);
    }

    fut_opencl_src = (char*) malloc(src_size + 1);

    size_t n, i;
    for (i = 0, n = 0; srcs && srcs[i]; i++) {
      strncpy(fut_opencl_src+n, srcs[i], src_size-n);
      n += strlen(srcs[i]);
    }
    fut_opencl_src[src_size] = 0;

  }

  cl_program prog;
  error = CL_SUCCESS;
  const char* src_ptr[] = {fut_opencl_src};

  if (ctx->cfg.dump_program_to != NULL) {
    FILE *f = fopen(ctx->cfg.dump_program_to, "w");
    assert(f != NULL);
    fputs(fut_opencl_src, f);
    fclose(f);
  }

  if (ctx->cfg.load_binary_from == NULL) {
    prog = clCreateProgramWithSource(ctx->ctx, 1, src_ptr, &src_size, &error);
    OPENCL_SUCCEED_FATAL(error);

    int compile_opts_size = 1024;

    for (int i = 0; i < ctx->cfg.num_sizes; i++) {
      compile_opts_size += strlen(ctx->cfg.size_names[i]) + 20;
    }

    for (int i = 0; extra_build_opts[i] != NULL; i++) {
      compile_opts_size += strlen(extra_build_opts[i] + 1);
    }

    char *compile_opts = (char*) malloc(compile_opts_size);

    int w = snprintf(compile_opts, compile_opts_size,
                     "-DLOCKSTEP_WIDTH=%d ",
                     (int)ctx->lockstep_width);

    for (int i = 0; i < ctx->cfg.num_sizes; i++) {
      w += snprintf(compile_opts+w, compile_opts_size-w,
                    "-D%s=%d ",
                    ctx->cfg.size_vars[i],
                    (int)ctx->cfg.size_values[i]);
    }

    for (int i = 0; extra_build_opts[i] != NULL; i++) {
      w += snprintf(compile_opts+w, compile_opts_size-w,
                    "%s ", extra_build_opts[i]);
    }

    OPENCL_SUCCEED_FATAL(build_opencl_program(prog, device_option.device, compile_opts));

    free(compile_opts);
  } else {
    size_t binary_size;
    unsigned char *fut_opencl_bin =
      (unsigned char*) slurp_file(ctx->cfg.load_binary_from, &binary_size);
    assert(fut_opencl_src != NULL);
    const unsigned char *binaries[1] = { fut_opencl_bin };
    cl_int status = 0;

    prog = clCreateProgramWithBinary(ctx->ctx, 1, &device_option.device,
                                     &binary_size, binaries,
                                     &status, &error);

    OPENCL_SUCCEED_FATAL(status);
    OPENCL_SUCCEED_FATAL(error);
  }

  free(fut_opencl_src);

  if (ctx->cfg.dump_binary_to != NULL) {
    size_t binary_size;
    OPENCL_SUCCEED_FATAL(clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES,
                                          sizeof(size_t), &binary_size, NULL));
    unsigned char *binary = (unsigned char*) malloc(binary_size);
    unsigned char *binaries[1] = { binary };
    OPENCL_SUCCEED_FATAL(clGetProgramInfo(prog, CL_PROGRAM_BINARIES,
                                          sizeof(unsigned char*), binaries, NULL));

    FILE *f = fopen(ctx->cfg.dump_binary_to, "w");
    assert(f != NULL);
    fwrite(binary, sizeof(char), binary_size, f);
    fclose(f);
  }

  return prog;
}

static cl_program setup_opencl(struct opencl_context *ctx,
                               const char *srcs[],
                               int required_types,
                               const char *extra_build_opts[]) {

  ctx->lockstep_width = 0; // Real value set later.

  free_list_init(&ctx->free_list);

  struct opencl_device_option device_option = get_preferred_device(&ctx->cfg);

  if (ctx->cfg.logging) {
    describe_device_option(device_option);
  }

  // Note that NVIDIA's OpenCL requires the platform property
  cl_context_properties properties[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)device_option.platform,
    0
  };

  cl_int clCreateContext_error;
  ctx->ctx = clCreateContext(properties, 1, &device_option.device, NULL, NULL, &clCreateContext_error);
  OPENCL_SUCCEED_FATAL(clCreateContext_error);

  cl_int clCreateCommandQueue_error;
  cl_command_queue queue =
    clCreateCommandQueue(ctx->ctx,
                         device_option.device,
                         ctx->cfg.profiling ? CL_QUEUE_PROFILING_ENABLE : 0,
                         &clCreateCommandQueue_error);
  OPENCL_SUCCEED_FATAL(clCreateCommandQueue_error);

  return setup_opencl_with_command_queue(ctx, queue, srcs, required_types, extra_build_opts);
}

// Count up the runtime all the profiling_records that occured during execution.
// Also clears the buffer of profiling_records.
static cl_int opencl_tally_profiling_records(struct opencl_context *ctx) {
  cl_int err;
  for (int i = 0; i < ctx->profiling_records_used; i++) {
    struct profiling_record record = ctx->profiling_records[i];

    cl_ulong start_t, end_t;

    if ((err = clGetEventProfilingInfo(*record.event,
                                       CL_PROFILING_COMMAND_START,
                                       sizeof(start_t),
                                       &start_t,
                                       NULL)) != CL_SUCCESS) {
      return err;
    }

    if ((err = clGetEventProfilingInfo(*record.event,
                                       CL_PROFILING_COMMAND_END,
                                       sizeof(end_t),
                                       &end_t,
                                       NULL)) != CL_SUCCESS) {
      return err;
    }

    // OpenCL provides nanosecond resolution, but we want
    // microseconds.
    *record.runs += 1;
    *record.runtime += (end_t - start_t)/1000;

    if ((err = clReleaseEvent(*record.event)) != CL_SUCCESS) {
      return err;
    }
    free(record.event);
  }

  ctx->profiling_records_used = 0;

  return CL_SUCCESS;
}

// If profiling, produce an event associated with a profiling record.
static cl_event* opencl_get_event(struct opencl_context *ctx, int *runs, int64_t *runtime) {
    if (ctx->profiling_records_used == ctx->profiling_records_capacity) {
      ctx->profiling_records_capacity *= 2;
      ctx->profiling_records =
        realloc(ctx->profiling_records,
                ctx->profiling_records_capacity *
                sizeof(struct profiling_record));
    }
    cl_event *event = malloc(sizeof(cl_event));
    ctx->profiling_records[ctx->profiling_records_used].event = event;
    ctx->profiling_records[ctx->profiling_records_used].runs = runs;
    ctx->profiling_records[ctx->profiling_records_used].runtime = runtime;
    ctx->profiling_records_used++;
    return event;
}

// Allocate memory from driver. The problem is that OpenCL may perform
// lazy allocation, so we cannot know whether an allocation succeeded
// until the first time we try to use it.  Hence we immediately
// perform a write to see if the allocation succeeded.  This is slow,
// but the assumption is that this operation will be rare (most things
// will go through the free list).
static int opencl_alloc_actual(struct opencl_context *ctx, size_t size, cl_mem *mem_out) {
  int error;
  *mem_out = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, size, NULL, &error);

  if (error != CL_SUCCESS) {
    return error;
  }

  int x = 2;
  error = clEnqueueWriteBuffer(ctx->queue, *mem_out, 1, 0, sizeof(x), &x, 0, NULL, NULL);

  // No need to wait for completion here. clWaitForEvents() cannot
  // return mem object allocation failures. This implies that the
  // buffer is faulted onto the device on enqueue. (Observation by
  // Andreas Kloeckner.)

  return error;
}

static int opencl_alloc(struct opencl_context *ctx, size_t min_size, const char *tag, cl_mem *mem_out) {
  if (min_size < sizeof(int)) {
    min_size = sizeof(int);
  }

  size_t size;

  if (free_list_find(&ctx->free_list, tag, &size, mem_out) == 0) {
    // Successfully found a free block.  Is it big enough?
    //
    // FIXME: we might also want to check whether the block is *too
    // big*, to avoid internal fragmentation.  However, this can
    // sharply impact performance on programs where arrays change size
    // frequently.  Fortunately, such allocations are usually fairly
    // short-lived, as they are necessarily within a loop, so the risk
    // of internal fragmentation resulting in an OOM situation is
    // limited.  However, it would be preferable if we could go back
    // and *shrink* oversize allocations when we encounter an OOM
    // condition.  That is technically feasible, since we do not
    // expose OpenCL pointer values directly to the application, but
    // instead rely on a level of indirection.
    if (size >= min_size) {
      return CL_SUCCESS;
    } else {
      // Not just right - free it.
      int error = clReleaseMemObject(*mem_out);
      if (error != CL_SUCCESS) {
        return error;
      }
    }
  }

  // We have to allocate a new block from the driver.  If the
  // allocation does not succeed, then we might be in an out-of-memory
  // situation.  We now start freeing things from the free list until
  // we think we have freed enough that the allocation will succeed.
  // Since we don't know how far the allocation is from fitting, we
  // have to check after every deallocation.  This might be pretty
  // expensive.  Let's hope that this case is hit rarely.

  int error = opencl_alloc_actual(ctx, min_size, mem_out);

  while (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
    if (ctx->cfg.debugging) {
      fprintf(stderr, "Out of OpenCL memory: releasing entry from the free list...\n");
    }
    cl_mem mem;
    if (free_list_first(&ctx->free_list, &mem) == 0) {
      error = clReleaseMemObject(mem);
      if (error != CL_SUCCESS) {
        return error;
      }
    } else {
      break;
    }
    error = opencl_alloc_actual(ctx, min_size, mem_out);
  }

  return error;
}

static int opencl_free(struct opencl_context *ctx, cl_mem mem, const char *tag) {
  size_t size;
  cl_mem existing_mem;

  // If there is already a block with this tag, then remove it.
  if (free_list_find(&ctx->free_list, tag, &size, &existing_mem) == 0) {
    int error = clReleaseMemObject(existing_mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  int error = clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(size_t), &size, NULL);

  if (error == CL_SUCCESS) {
    free_list_insert(&ctx->free_list, size, mem, tag);
  }

  return error;
}

static int opencl_free_all(struct opencl_context *ctx) {
  cl_mem mem;
  free_list_pack(&ctx->free_list);
  while (free_list_first(&ctx->free_list, &mem) == 0) {
    int error = clReleaseMemObject(mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  return CL_SUCCESS;
}

// End of opencl.h.

static const char *opencl_program[] =
                  {"#ifdef cl_clang_storage_class_specifiers\n#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable\n#endif\n#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n__kernel void dummy_kernel(__global unsigned char *dummy, int n)\n{\n    const int thread_gid = get_global_id(0);\n    \n    if (thread_gid >= n)\n        return;\n}\ntypedef char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long int64_t;\ntypedef uchar uint8_t;\ntypedef ushort uint16_t;\ntypedef uint uint32_t;\ntypedef ulong uint64_t;\n#ifdef cl_nv_pragma_unroll\nstatic inline void mem_fence_global()\n{\n    asm(\"membar.gl;\");\n}\n#else\nstatic inline void mem_fence_global()\n{\n    mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n}\n#endif\nstatic inline void mem_fence_local()\n{\n    mem_fence(CLK_LOCAL_MEM_FENCE);\n}\nstatic inline int8_t add8(int8_t x, int8_t y)\n{\n    return x + y;\n}\nstatic inline int16_t add16(int16_t x, int16_t y)\n{\n    return x + y;\n}\nstatic inline int32_t add32(int32_t x, int32_t y)\n{\n    return x + y;\n}\nstatic inline int64_t add64(int64_t x, int64_t y)\n{\n    return x + y;\n}\nstatic inline int8_t sub8(int8_t x, int8_t y)\n{\n    return x - y;\n}\nstatic inline int16_t sub16(int16_t x, int16_t y)\n{\n    return x - y;\n}\nstatic inline int32_t sub32(int32_t x, int32_t y)\n{\n    return x - y;\n}\nstatic inline int64_t sub64(int64_t x, int64_t y)\n{\n    return x - y;\n}\nstatic inline int8_t mul8(int8_t x, int8_t y)\n{\n    return x * y;\n}\nstatic inline int16_t mul16(int16_t x, int16_t y)\n{\n    return x * y;\n}\nstatic inline int32_t mul32(int32_t x, int32_t y)\n{\n    return x * y;\n}\nstatic inline int64_t mul64(int64_t x, int64_t y)\n{\n    return x * y;\n}\nstatic inline uint8_t udiv8(uint8_t x, uint8_t y)\n{\n    return x / y;\n}\nstatic inline uint16_t udiv16(uint16_t x, uint16_t y)\n{\n    return x / y;\n}\nstatic inline uint32_t udiv32(uint32_t x, uint32_t y)\n{\n    return x / y;\n}\nstatic inline uint64_t udiv64(uint64_t x, uint64_t y)\n{\n    return x / y;\n}\nstatic inline uint8_t umod8(uint8_t x, uint",
                   "8_t y)\n{\n    return x % y;\n}\nstatic inline uint16_t umod16(uint16_t x, uint16_t y)\n{\n    return x % y;\n}\nstatic inline uint32_t umod32(uint32_t x, uint32_t y)\n{\n    return x % y;\n}\nstatic inline uint64_t umod64(uint64_t x, uint64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t sdiv8(int8_t x, int8_t y)\n{\n    int8_t q = x / y;\n    int8_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int16_t sdiv16(int16_t x, int16_t y)\n{\n    int16_t q = x / y;\n    int16_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int32_t sdiv32(int32_t x, int32_t y)\n{\n    int32_t q = x / y;\n    int32_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int64_t sdiv64(int64_t x, int64_t y)\n{\n    int64_t q = x / y;\n    int64_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int8_t smod8(int8_t x, int8_t y)\n{\n    int8_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int16_t smod16(int16_t x, int16_t y)\n{\n    int16_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int32_t smod32(int32_t x, int32_t y)\n{\n    int32_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int64_t smod64(int64_t x, int64_t y)\n{\n    int64_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int8_t squot8(int8_t x, int8_t y)\n{\n    return x / y;\n}\nstatic inline int16_t squot16(int16_t x, int16_t y)\n{\n    return x / y;\n}\nstatic inline int32_t squot32(int32_t x, int32_t y)\n{\n    return x / y;\n}\nstatic inline int64_t squot64(int64_t x, int64_t y)\n{\n    return x / y;\n}\nstatic inline int8_t srem8(int8_t x, int8_t y)\n{\n    return x % y;\n}\nstatic inline int16_t srem16(int16_t x, int16_t y)\n{\n    return x % y;\n}\nstatic inline int32_t srem32(int32_t x, int32_t y)\n{\n    ret",
                   "urn x % y;\n}\nstatic inline int64_t srem64(int64_t x, int64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t smin8(int8_t x, int8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int16_t smin16(int16_t x, int16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int32_t smin32(int32_t x, int32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int64_t smin64(int64_t x, int64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint8_t umin8(uint8_t x, uint8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint16_t umin16(uint16_t x, uint16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint32_t umin32(uint32_t x, uint32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint64_t umin64(uint64_t x, uint64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int8_t smax8(int8_t x, int8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int16_t smax16(int16_t x, int16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int32_t smax32(int32_t x, int32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int64_t smax64(int64_t x, int64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t umax8(uint8_t x, uint8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint16_t umax16(uint16_t x, uint16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint32_t umax32(uint32_t x, uint32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint64_t umax64(uint64_t x, uint64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t shl8(uint8_t x, uint8_t y)\n{\n    return x << y;\n}\nstatic inline uint16_t shl16(uint16_t x, uint16_t y)\n{\n    return x << y;\n}\nstatic inline uint32_t shl32(uint32_t x, uint32_t y)\n{\n    return x << y;\n}\nstatic inline uint64_t shl64(uint64_t x, uint64_t y)\n{\n    return x << y;\n}\nstatic inline uint8_t lshr8(uint8_t x, uint8_t y)\n{\n    return x >> y;\n}\nstatic inline uint16_t lshr16(uint16_t x, uint16_t y)\n{\n    return x >> y;\n}\nstatic inline uint32_t lshr32(uint32_t x, uint32_t y)\n{\n    return x >> y;\n}\nstatic inline uint64_t lshr64(uint64_t x, uint64_t y)\n{\n    return x >> y;\n}\nstatic",
                   " inline int8_t ashr8(int8_t x, int8_t y)\n{\n    return x >> y;\n}\nstatic inline int16_t ashr16(int16_t x, int16_t y)\n{\n    return x >> y;\n}\nstatic inline int32_t ashr32(int32_t x, int32_t y)\n{\n    return x >> y;\n}\nstatic inline int64_t ashr64(int64_t x, int64_t y)\n{\n    return x >> y;\n}\nstatic inline uint8_t and8(uint8_t x, uint8_t y)\n{\n    return x & y;\n}\nstatic inline uint16_t and16(uint16_t x, uint16_t y)\n{\n    return x & y;\n}\nstatic inline uint32_t and32(uint32_t x, uint32_t y)\n{\n    return x & y;\n}\nstatic inline uint64_t and64(uint64_t x, uint64_t y)\n{\n    return x & y;\n}\nstatic inline uint8_t or8(uint8_t x, uint8_t y)\n{\n    return x | y;\n}\nstatic inline uint16_t or16(uint16_t x, uint16_t y)\n{\n    return x | y;\n}\nstatic inline uint32_t or32(uint32_t x, uint32_t y)\n{\n    return x | y;\n}\nstatic inline uint64_t or64(uint64_t x, uint64_t y)\n{\n    return x | y;\n}\nstatic inline uint8_t xor8(uint8_t x, uint8_t y)\n{\n    return x ^ y;\n}\nstatic inline uint16_t xor16(uint16_t x, uint16_t y)\n{\n    return x ^ y;\n}\nstatic inline uint32_t xor32(uint32_t x, uint32_t y)\n{\n    return x ^ y;\n}\nstatic inline uint64_t xor64(uint64_t x, uint64_t y)\n{\n    return x ^ y;\n}\nstatic inline char ult8(uint8_t x, uint8_t y)\n{\n    return x < y;\n}\nstatic inline char ult16(uint16_t x, uint16_t y)\n{\n    return x < y;\n}\nstatic inline char ult32(uint32_t x, uint32_t y)\n{\n    return x < y;\n}\nstatic inline char ult64(uint64_t x, uint64_t y)\n{\n    return x < y;\n}\nstatic inline char ule8(uint8_t x, uint8_t y)\n{\n    return x <= y;\n}\nstatic inline char ule16(uint16_t x, uint16_t y)\n{\n    return x <= y;\n}\nstatic inline char ule32(uint32_t x, uint32_t y)\n{\n    return x <= y;\n}\nstatic inline char ule64(uint64_t x, uint64_t y)\n{\n    return x <= y;\n}\nstatic inline char slt8(int8_t x, int8_t y)\n{\n    return x < y;\n}\nstatic inline char slt16(int16_t x, int16_t y)\n{\n    return x < y;\n}\nstatic inline char slt32(int32_t x, int32_t y)\n{\n    return x < y;\n}\nstatic inline char slt64(int64_t x, int64_t y)\n{\n    return ",
                   "x < y;\n}\nstatic inline char sle8(int8_t x, int8_t y)\n{\n    return x <= y;\n}\nstatic inline char sle16(int16_t x, int16_t y)\n{\n    return x <= y;\n}\nstatic inline char sle32(int32_t x, int32_t y)\n{\n    return x <= y;\n}\nstatic inline char sle64(int64_t x, int64_t y)\n{\n    return x <= y;\n}\nstatic inline int8_t pow8(int8_t x, int8_t y)\n{\n    int8_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int16_t pow16(int16_t x, int16_t y)\n{\n    int16_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int32_t pow32(int32_t x, int32_t y)\n{\n    int32_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int64_t pow64(int64_t x, int64_t y)\n{\n    int64_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline bool itob_i8_bool(int8_t x)\n{\n    return x;\n}\nstatic inline bool itob_i16_bool(int16_t x)\n{\n    return x;\n}\nstatic inline bool itob_i32_bool(int32_t x)\n{\n    return x;\n}\nstatic inline bool itob_i64_bool(int64_t x)\n{\n    return x;\n}\nstatic inline int8_t btoi_bool_i8(bool x)\n{\n    return x;\n}\nstatic inline int16_t btoi_bool_i16(bool x)\n{\n    return x;\n}\nstatic inline int32_t btoi_bool_i32(bool x)\n{\n    return x;\n}\nstatic inline int64_t btoi_bool_i64(bool x)\n{\n    return x;\n}\n#define sext_i8_i8(x) ((int8_t) (int8_t) x)\n#define sext_i8_i16(x) ((int16_t) (int8_t) x)\n#define sext_i8_i32(x) ((int32_t) (int8_t) x)\n#define sext_i8_i64(x) ((int64_t) (int8_t) x)\n#define sext_i16_i8(x) ((int8_t) (int16_t) x)\n#define sext_i16_i16(x) ((int16_t) (int16_t) x)\n#define sext_i16_i32(x) ((int32_t) (int16_t) x)\n#define sext_i16_i64(x) ((int64_t) (int16_t) x)\n#define",
                   " sext_i32_i8(x) ((int8_t) (int32_t) x)\n#define sext_i32_i16(x) ((int16_t) (int32_t) x)\n#define sext_i32_i32(x) ((int32_t) (int32_t) x)\n#define sext_i32_i64(x) ((int64_t) (int32_t) x)\n#define sext_i64_i8(x) ((int8_t) (int64_t) x)\n#define sext_i64_i16(x) ((int16_t) (int64_t) x)\n#define sext_i64_i32(x) ((int32_t) (int64_t) x)\n#define sext_i64_i64(x) ((int64_t) (int64_t) x)\n#define zext_i8_i8(x) ((uint8_t) (uint8_t) x)\n#define zext_i8_i16(x) ((uint16_t) (uint8_t) x)\n#define zext_i8_i32(x) ((uint32_t) (uint8_t) x)\n#define zext_i8_i64(x) ((uint64_t) (uint8_t) x)\n#define zext_i16_i8(x) ((uint8_t) (uint16_t) x)\n#define zext_i16_i16(x) ((uint16_t) (uint16_t) x)\n#define zext_i16_i32(x) ((uint32_t) (uint16_t) x)\n#define zext_i16_i64(x) ((uint64_t) (uint16_t) x)\n#define zext_i32_i8(x) ((uint8_t) (uint32_t) x)\n#define zext_i32_i16(x) ((uint16_t) (uint32_t) x)\n#define zext_i32_i32(x) ((uint32_t) (uint32_t) x)\n#define zext_i32_i64(x) ((uint64_t) (uint32_t) x)\n#define zext_i64_i8(x) ((uint8_t) (uint64_t) x)\n#define zext_i64_i16(x) ((uint16_t) (uint64_t) x)\n#define zext_i64_i32(x) ((uint32_t) (uint64_t) x)\n#define zext_i64_i64(x) ((uint64_t) (uint64_t) x)\n#ifdef __OPENCL_VERSION__\nint32_t futrts_popc8(int8_t x)\n{\n    return popcount(x);\n}\nint32_t futrts_popc16(int16_t x)\n{\n    return popcount(x);\n}\nint32_t futrts_popc32(int32_t x)\n{\n    return popcount(x);\n}\nint32_t futrts_popc64(int64_t x)\n{\n    return popcount(x);\n}\n#elif __CUDA_ARCH__\nint32_t futrts_popc8(int8_t x)\n{\n    return __popc(zext_i8_i32(x));\n}\nint32_t futrts_popc16(int16_t x)\n{\n    return __popc(zext_i16_i32(x));\n}\nint32_t futrts_popc32(int32_t x)\n{\n    return __popc(x);\n}\nint32_t futrts_popc64(int64_t x)\n{\n    return __popcll(x);\n}\n#else\nint32_t futrts_popc8(int8_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nint32_t futrts_popc16(int16_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nint32_t futrts_popc32(int32_t x)\n{\n    int c = 0;\n    \n    for (; x;",
                   " ++c)\n        x &= x - 1;\n    return c;\n}\nint32_t futrts_popc64(int64_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\n#endif\n#ifdef __OPENCL_VERSION__\nint32_t futrts_clzz8(int8_t x)\n{\n    return clz(x);\n}\nint32_t futrts_clzz16(int16_t x)\n{\n    return clz(x);\n}\nint32_t futrts_clzz32(int32_t x)\n{\n    return clz(x);\n}\nint32_t futrts_clzz64(int64_t x)\n{\n    return clz(x);\n}\n#elif __CUDA_ARCH__\nint32_t futrts_clzz8(int8_t x)\n{\n    return __clz(zext_i8_i32(x)) - 24;\n}\nint32_t futrts_clzz16(int16_t x)\n{\n    return __clz(zext_i16_i32(x)) - 16;\n}\nint32_t futrts_clzz32(int32_t x)\n{\n    return __clz(x);\n}\nint32_t futrts_clzz64(int64_t x)\n{\n    return __clzll(x);\n}\n#else\nint32_t futrts_clzz8(int8_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nint32_t futrts_clzz16(int16_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nint32_t futrts_clzz32(int32_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nint32_t futrts_clzz64(int64_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\n#endif\nstatic inline float fdiv32(float x, float y)\n{\n    return x / y;\n}\nstatic inline float fadd32(float x, float y)\n{\n    return x + y;\n}\nstatic inline float fsub32(float x, float y)\n{\n    return x - y;\n}\nstatic inline float fmul32(float x, float y)\n{\n    return x * y;\n}\nstatic inline float fmin32(float x, float y)\n{\n    return fmin(x, y);\n}\nstatic inline float fmax32(float x, float y)\n{\n    return fmax(x, y);\n}\nstatic inline float fpow32(float ",
                   "x, float y)\n{\n    return pow(x, y);\n}\nstatic inline char cmplt32(float x, float y)\n{\n    return x < y;\n}\nstatic inline char cmple32(float x, float y)\n{\n    return x <= y;\n}\nstatic inline float sitofp_i8_f32(int8_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i16_f32(int16_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i32_f32(int32_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i64_f32(int64_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i8_f32(uint8_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i16_f32(uint16_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i32_f32(uint32_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i64_f32(uint64_t x)\n{\n    return x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float futrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n{\n    return atan(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2",
                   "(x, y);\n}\nstatic inline float futrts_gamma32(float x)\n{\n    return tgamma(x);\n}\nstatic inline float futrts_lgamma32(float x)\n{\n    return lgamma(x);\n}\nstatic inline char futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline char futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#ifdef __OPENCL_VERSION__\nstatic inline float fmod32(float x, float y)\n{\n    return fmod(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floor(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceil(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t)\n{\n    return mix(v0, v1, t);\n}\n#else\nstatic inline float fmod32(float x, float y)\n{\n    return fmodf(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rintf(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floorf(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceilf(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t)\n{\n    return v0 + (v1 - v0) * t;\n}\n#endif\n__kernel void copy_10149(int32_t sizze_9183, int32_t sizze_9184,\n                         int32_t conc_tmp_9220, __global\n                         unsigned char *xs_mem_10004, __global\n                         unsigned char *mem_10055, int32_t tmp_offs_10148)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t copy_gtid_10149;\n    int32_t copy_ltid_10150;\n    int32_t copy_gid_10151;\n    \n    copy_gtid_10149 = get_global_id(0);\n    copy_ltid_10150 = get_local_id(0);\n    copy_gid_10151 = get_group_id(0);\n    if (slt32(copy_gtid_10149, sizze_9183 * (sizze_9184 * sizze_9184))) {\n    ",
                   "    ((__global float *) mem_10055)[conc_tmp_9220 * sizze_9184 * 0 +\n                                       conc_tmp_9220 * 0 + tmp_offs_10148 +\n                                       (squot32(copy_gtid_10149, sizze_9184 *\n                                                sizze_9184) * (conc_tmp_9220 *\n                                                               sizze_9184) +\n                                        squot32(copy_gtid_10149 -\n                                                squot32(copy_gtid_10149,\n                                                        sizze_9184 *\n                                                        sizze_9184) *\n                                                (sizze_9184 * sizze_9184),\n                                                sizze_9184) * conc_tmp_9220 +\n                                        (copy_gtid_10149 -\n                                         squot32(copy_gtid_10149, sizze_9184 *\n                                                 sizze_9184) * (sizze_9184 *\n                                                                sizze_9184) -\n                                         squot32(copy_gtid_10149 -\n                                                 squot32(copy_gtid_10149,\n                                                         sizze_9184 *\n                                                         sizze_9184) *\n                                                 (sizze_9184 * sizze_9184),\n                                                 sizze_9184) * sizze_9184))] =\n            ((__global float *) xs_mem_10004)[squot32(copy_gtid_10149,\n                                                      sizze_9184 * sizze_9184) *\n                                              (sizze_9184 * sizze_9184) +\n                                              squot32(copy_gtid_10149 -\n                                                      squot32(copy_gtid_10149,\n                                                              sizze_9184 *\n         ",
                   "                                                     sizze_9184) *\n                                                      (sizze_9184 * sizze_9184),\n                                                      sizze_9184) * sizze_9184 +\n                                              (copy_gtid_10149 -\n                                               squot32(copy_gtid_10149,\n                                                       sizze_9184 *\n                                                       sizze_9184) *\n                                               (sizze_9184 * sizze_9184) -\n                                               squot32(copy_gtid_10149 -\n                                                       squot32(copy_gtid_10149,\n                                                               sizze_9184 *\n                                                               sizze_9184) *\n                                                       (sizze_9184 *\n                                                        sizze_9184),\n                                                       sizze_9184) *\n                                               sizze_9184)];\n    }\n}\n__kernel void copy_10154(int32_t sizze_9183, int32_t sizze_9184,\n                         int32_t conc_tmp_9220, __global\n                         unsigned char *mem_10048, __global\n                         unsigned char *mem_10055, int32_t tmp_offs_10148)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t copy_gtid_10154;\n    int32_t copy_ltid_10155;\n    int32_t copy_gid_10156;\n    \n    copy_gtid_10154 = get_global_id(0);\n    copy_ltid_10155 = get_local_id(0);\n    copy_gid_10156 = get_group_id(0);\n    if (slt32(copy_gtid_10154, sizze_9183 * (sizze_9184 * sizze_9184))) {\n        ((__global float *) mem_10055)[conc_tmp_9220 * sizze_9184 * 0 +\n                                       conc_tmp_9220 * 0 + tmp_offs_10148 +\n                                       (squot32(copy_gtid_101",
                   "54, sizze_9184 *\n                                                sizze_9184) * (conc_tmp_9220 *\n                                                               sizze_9184) +\n                                        squot32(copy_gtid_10154 -\n                                                squot32(copy_gtid_10154,\n                                                        sizze_9184 *\n                                                        sizze_9184) *\n                                                (sizze_9184 * sizze_9184),\n                                                sizze_9184) * conc_tmp_9220 +\n                                        (copy_gtid_10154 -\n                                         squot32(copy_gtid_10154, sizze_9184 *\n                                                 sizze_9184) * (sizze_9184 *\n                                                                sizze_9184) -\n                                         squot32(copy_gtid_10154 -\n                                                 squot32(copy_gtid_10154,\n                                                         sizze_9184 *\n                                                         sizze_9184) *\n                                                 (sizze_9184 * sizze_9184),\n                                                 sizze_9184) * sizze_9184))] =\n            ((__global float *) mem_10048)[squot32(copy_gtid_10154, sizze_9184 *\n                                                   sizze_9184) * (sizze_9184 *\n                                                                  sizze_9184) +\n                                           squot32(copy_gtid_10154 -\n                                                   squot32(copy_gtid_10154,\n                                                           sizze_9184 *\n                                                           sizze_9184) *\n                                                   (sizze_9184 * sizze_9184),\n                                                   sizze_9",
                   "184) * sizze_9184 +\n                                           (copy_gtid_10154 -\n                                            squot32(copy_gtid_10154,\n                                                    sizze_9184 * sizze_9184) *\n                                            (sizze_9184 * sizze_9184) -\n                                            squot32(copy_gtid_10154 -\n                                                    squot32(copy_gtid_10154,\n                                                            sizze_9184 *\n                                                            sizze_9184) *\n                                                    (sizze_9184 * sizze_9184),\n                                                    sizze_9184) * sizze_9184)];\n    }\n}\n__kernel void copy_10171(int32_t sizze_9183, int32_t sizze_9184,\n                         int32_t conc_tmp_9220, __global\n                         unsigned char *res_r_mem_10078, __global\n                         unsigned char *mem_10085)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t copy_gtid_10171;\n    int32_t copy_ltid_10172;\n    int32_t copy_gid_10173;\n    \n    copy_gtid_10171 = get_global_id(0);\n    copy_ltid_10172 = get_local_id(0);\n    copy_gid_10173 = get_group_id(0);\n    if (slt32(copy_gtid_10171, sizze_9183 * (sizze_9184 * sizze_9184))) {\n        ((__global float *) mem_10085)[squot32(copy_gtid_10171, sizze_9184 *\n                                               sizze_9184) * (sizze_9184 *\n                                                              sizze_9184) +\n                                       squot32(copy_gtid_10171 -\n                                               squot32(copy_gtid_10171,\n                                                       sizze_9184 *\n                                                       sizze_9184) *\n                                               (sizze_9184 * sizze_9184),\n                                               s",
                   "izze_9184) * sizze_9184 +\n                                       (copy_gtid_10171 -\n                                        squot32(copy_gtid_10171, sizze_9184 *\n                                                sizze_9184) * (sizze_9184 *\n                                                               sizze_9184) -\n                                        squot32(copy_gtid_10171 -\n                                                squot32(copy_gtid_10171,\n                                                        sizze_9184 *\n                                                        sizze_9184) *\n                                                (sizze_9184 * sizze_9184),\n                                                sizze_9184) * sizze_9184)] =\n            ((__global float *) res_r_mem_10078)[sizze_9184 +\n                                                 (squot32(copy_gtid_10171,\n                                                          sizze_9184 *\n                                                          sizze_9184) *\n                                                  (conc_tmp_9220 * sizze_9184) +\n                                                  squot32(copy_gtid_10171 -\n                                                          squot32(copy_gtid_10171,\n                                                                  sizze_9184 *\n                                                                  sizze_9184) *\n                                                          (sizze_9184 *\n                                                           sizze_9184),\n                                                          sizze_9184) *\n                                                  conc_tmp_9220 +\n                                                  (copy_gtid_10171 -\n                                                   squot32(copy_gtid_10171,\n                                                           sizze_9184 *\n                                                           sizze_9184) *\n    ",
                   "                                               (sizze_9184 * sizze_9184) -\n                                                   squot32(copy_gtid_10171 -\n                                                           squot32(copy_gtid_10171,\n                                                                   sizze_9184 *\n                                                                   sizze_9184) *\n                                                           (sizze_9184 *\n                                                            sizze_9184),\n                                                           sizze_9184) *\n                                                   sizze_9184))];\n    }\n}\n__kernel void map_transpose_f32(__local volatile\n                                int64_t *block_11_backing_aligned_0,\n                                int32_t destoffset_1, int32_t srcoffset_3,\n                                int32_t num_arrays_4, int32_t x_elems_5,\n                                int32_t y_elems_6, int32_t in_elems_7,\n                                int32_t out_elems_8, int32_t mulx_9,\n                                int32_t muly_10, __global\n                                unsigned char *destmem_0, __global\n                                unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_11_backing_0 = (__local volatile\n                                                          char *) block_11_backing_aligned_0;\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    ",
                   "get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;\n    int32_t x_index_31 = get_global_id_0_37;\n    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;\n    \n    if (slt32(x_index_31, x_elems_5)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +\n                    x_index_31;\n            \n            if (slt32(y_index_32 + j_43 * 8, y_elems_6) && slt32(index_in_35,\n                                                                 in_elems_7)) {\n                ((__local float *) block_11)[(get_local_id_1_39 + j_43 * 8) *\n                                             33 + get_local_id_0_38] =\n                    ((__global float *) srcmem_2)[idata_offset_34 +\n                                                  index_in_35];\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;\n    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;\n    if (slt32(x_index_31, y_elems_6)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +\n                    x_index_31;\n            \n            if (slt32(y_index_32 + j_43 * 8, x_elems_5) && slt32(index_out_36,\n                                                                 out_elems_8)) {\n                ((__global float *) destmem_0)[odata_offset_33 + index_out_36] =\n                    ((__local float *) block_11)[get_local_id_0_38 * 33 +\n                                                 get_local_id_1_39 + j_43 * 8];\n            }\n        }\n    }\n}\n__kernel void map_transpose_f32_low_height(__local volat",
                   "ile\n                                           int64_t *block_11_backing_aligned_0,\n                                           int32_t destoffset_1,\n                                           int32_t srcoffset_3,\n                                           int32_t num_arrays_4,\n                                           int32_t x_elems_5, int32_t y_elems_6,\n                                           int32_t in_elems_7,\n                                           int32_t out_elems_8, int32_t mulx_9,\n                                           int32_t muly_10, __global\n                                           unsigned char *destmem_0, __global\n                                           unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_11_backing_0 = (__local volatile\n                                                          char *) block_11_backing_aligned_0;\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;\n    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_0_38 +\n            srem32(get_local_id_1_39, mulx_9) * 16;\n    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_3",
                   "9,\n                                                          mulx_9);\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    \n    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&\n                                         slt32(index_in_35, in_elems_7))) {\n        ((__local float *) block_11)[get_local_id_1_39 * 17 +\n                                     get_local_id_0_38] = ((__global\n                                                            float *) srcmem_2)[idata_offset_34 +\n                                                                               index_in_35];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_9);\n    y_index_32 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_1_39 +\n        srem32(get_local_id_0_38, mulx_9) * 16;\n    \n    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;\n    \n    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&\n                                         slt32(index_out_36, out_elems_8))) {\n        ((__global float *) destmem_0)[odata_offset_33 + index_out_36] =\n            ((__local float *) block_11)[get_local_id_0_38 * 17 +\n                                         get_local_id_1_39];\n    }\n}\n__kernel void map_transpose_f32_low_width(__local volatile\n                                          int64_t *block_11_backing_aligned_0,\n                                          int32_t destoffset_1,\n                                          int32_t srcoffset_3,\n                                          int32_t num_arrays_4,\n                                          int32_t x_elems_5, int32_t y_elems_6,\n                                          int32_t in_elems_7,\n                                          int32_t out_elems_8, int32_t mulx_9,\n                                          int32_t muly_10, __global\n                                          unsigned char *destmem_0, __global\n                           ",
                   "               unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_11_backing_0 = (__local volatile\n                                                          char *) block_11_backing_aligned_0;\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;\n    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,\n                                                          muly_10);\n    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_10 + get_local_id_1_39 +\n            srem32(get_local_id_0_38, muly_10) * 16;\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    \n    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&\n                                         slt32(index_in_35, in_elems_7))) {\n        ((__local float *) block_11)[get_local_id_1_39 * 17 +\n                                     get_local_id_0_38] = ((__global\n                                                            float *) srcmem_2)[idata_offset_34 +\n                                                                               index_in_35];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group",
                   "_id_1_41 * 16 * muly_10 + get_local_id_0_38 +\n        srem32(get_local_id_1_39, muly_10) * 16;\n    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_10);\n    \n    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;\n    \n    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&\n                                         slt32(index_out_36, out_elems_8))) {\n        ((__global float *) destmem_0)[odata_offset_33 + index_out_36] =\n            ((__local float *) block_11)[get_local_id_0_38 * 17 +\n                                         get_local_id_1_39];\n    }\n}\n__kernel void map_transpose_f32_small(__local volatile\n                                      int64_t *block_11_backing_aligned_0,\n                                      int32_t destoffset_1, int32_t srcoffset_3,\n                                      int32_t num_arrays_4, int32_t x_elems_5,\n                                      int32_t y_elems_6, int32_t in_elems_7,\n                                      int32_t out_elems_8, int32_t mulx_9,\n                                      int32_t muly_10, __global\n                                      unsigned char *destmem_0, __global\n                                      unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_11_backing_0 = (__local volatile\n                                                          char *) block_11_backing_aligned_0;\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group",
                   "_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *\n                                          x_elems_5) * (y_elems_6 * x_elems_5);\n    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *\n                                        x_elems_5), y_elems_6);\n    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);\n    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;\n    \n    if (slt32(get_global_id_0_37, in_elems_7)) {\n        ((__global float *) destmem_0)[odata_offset_33 + index_out_36] =\n            ((__global float *) srcmem_2)[idata_offset_34 + index_in_35];\n    }\n}\n__kernel void replicate_10119(__global unsigned char *mem_10115,\n                              int32_t num_elems_10116, float val_10117)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_10119;\n    int32_t replicate_ltid_10120;\n    int32_t replicate_gid_10121;\n    \n    replicate_gtid_10119 = get_global_id(0);\n    replicate_ltid_10120 = get_local_id(0);\n    replicate_gid_10121 = get_group_id(0);\n    if (slt32(replicate_gtid_10119, num_elems_10116)) {\n        ((__global float *) mem_10115)[replicate_gtid_10119] = val_10117;\n    }\n}\n__kernel void replicate_10143(int32_t sizze_9183, int32_t sizze_9184, __global\n                              unsigned char *mem_10009, __global\n                              unsigned char *mem_10048)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_10143;\n    int32_t replicate_ltid_10144;\n    int32_t replicate_gid_10145;\n    \n    replicate_gtid_10143 = get_global_id(0);\n    replicat",
                   "e_ltid_10144 = get_local_id(0);\n    replicate_gid_10145 = get_group_id(0);\n    if (slt32(replicate_gtid_10143, sizze_9183 * sizze_9184 * sizze_9184)) {\n        ((__global float *) mem_10048)[squot32(replicate_gtid_10143,\n                                               sizze_9184 * sizze_9184) *\n                                       (sizze_9184 * sizze_9184) +\n                                       squot32(replicate_gtid_10143 -\n                                               squot32(replicate_gtid_10143,\n                                                       sizze_9184 *\n                                                       sizze_9184) *\n                                               (sizze_9184 * sizze_9184),\n                                               sizze_9184) * sizze_9184 +\n                                       (replicate_gtid_10143 -\n                                        squot32(replicate_gtid_10143,\n                                                sizze_9184 * sizze_9184) *\n                                        (sizze_9184 * sizze_9184) -\n                                        squot32(replicate_gtid_10143 -\n                                                squot32(replicate_gtid_10143,\n                                                        sizze_9184 *\n                                                        sizze_9184) *\n                                                (sizze_9184 * sizze_9184),\n                                                sizze_9184) * sizze_9184)] =\n            ((__global float *) mem_10009)[squot32(replicate_gtid_10143 -\n                                                   squot32(replicate_gtid_10143,\n                                                           sizze_9184 *\n                                                           sizze_9184) *\n                                                   (sizze_9184 * sizze_9184),\n                                                   sizze_9184) * sizze_9184 +\n                                  ",
                   "         (replicate_gtid_10143 -\n                                            squot32(replicate_gtid_10143,\n                                                    sizze_9184 * sizze_9184) *\n                                            (sizze_9184 * sizze_9184) -\n                                            squot32(replicate_gtid_10143 -\n                                                    squot32(replicate_gtid_10143,\n                                                            sizze_9184 *\n                                                            sizze_9184) *\n                                                    (sizze_9184 * sizze_9184),\n                                                    sizze_9184) * sizze_9184)];\n    }\n}\n__kernel void segmap_9302(int32_t sizze_9184, __global unsigned char *mem_10009)\n{\n    const int32_t segmap_group_sizze_9336 = mainzisegmap_group_sizze_9307;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_10124;\n    int32_t local_tid_10125;\n    int32_t group_sizze_10128;\n    int32_t wave_sizze_10127;\n    int32_t group_tid_10126;\n    \n    global_tid_10124 = get_global_id(0);\n    local_tid_10125 = get_local_id(0);\n    group_sizze_10128 = get_local_size(0);\n    wave_sizze_10127 = LOCKSTEP_WIDTH;\n    group_tid_10126 = get_group_id(0);\n    \n    int32_t phys_tid_9302 = global_tid_10124;\n    int32_t gtid_9300 = group_tid_10126 * segmap_group_sizze_9336 +\n            local_tid_10125;\n    int32_t gtid_9301;\n    \n    gtid_9301 = group_tid_10126 * segmap_group_sizze_9336 + local_tid_10125 -\n        (group_tid_10126 * segmap_group_sizze_9336 + local_tid_10125);\n    if (slt32(gtid_9300, sizze_9184) && slt32(gtid_9301, 1)) {\n        if ((sle32(0, gtid_9300) && slt32(gtid_9300, sizze_9184)) && (sle32(0,\n                                                                            gtid_9300) &&\n                                                                      slt32(gtid_9300,\n                        ",
                   "                                                    sizze_9184))) {\n            ((__global float *) mem_10009)[gtid_9300 * sizze_9184 + gtid_9300] =\n                1.0F;\n        }\n    }\n}\n__kernel void segmap_9609(int32_t sizze_9183, int32_t sizze_9184,\n                          int32_t conc_tmp_9220, int32_t y_9236, int32_t i_9862,\n                          __global unsigned char *A_expanded_mem_10056, __global\n                          unsigned char *mem_10070, __global\n                          unsigned char *mem_10077)\n{\n    const int32_t segmap_group_sizze_9930 = mainzisegmap_group_sizze_9616;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_10166;\n    int32_t local_tid_10167;\n    int32_t group_sizze_10170;\n    int32_t wave_sizze_10169;\n    int32_t group_tid_10168;\n    \n    global_tid_10166 = get_global_id(0);\n    local_tid_10167 = get_local_id(0);\n    group_sizze_10170 = get_local_size(0);\n    wave_sizze_10169 = LOCKSTEP_WIDTH;\n    group_tid_10168 = get_group_id(0);\n    \n    int32_t phys_tid_9609 = global_tid_10166;\n    int32_t gtid_9606 = squot32(group_tid_10168 * segmap_group_sizze_9930 +\n                                local_tid_10167, sizze_9184 * conc_tmp_9220);\n    int32_t gtid_9607 = squot32(group_tid_10168 * segmap_group_sizze_9930 +\n                                local_tid_10167 - squot32(group_tid_10168 *\n                                                          segmap_group_sizze_9930 +\n                                                          local_tid_10167,\n                                                          sizze_9184 *\n                                                          conc_tmp_9220) *\n                                (sizze_9184 * conc_tmp_9220), conc_tmp_9220);\n    int32_t gtid_9608;\n    \n    gtid_9608 = group_tid_10168 * segmap_group_sizze_9930 + local_tid_10167 -\n        squot32(group_tid_10168 * segmap_group_sizze_9930 + local_tid_10167,\n                sizze_918",
                   "4 * conc_tmp_9220) * (sizze_9184 * conc_tmp_9220) -\n        squot32(group_tid_10168 * segmap_group_sizze_9930 + local_tid_10167 -\n                squot32(group_tid_10168 * segmap_group_sizze_9930 +\n                        local_tid_10167, sizze_9184 * conc_tmp_9220) *\n                (sizze_9184 * conc_tmp_9220), conc_tmp_9220) * conc_tmp_9220;\n    if ((slt32(gtid_9606, sizze_9183) && slt32(gtid_9607, sizze_9184)) &&\n        slt32(gtid_9608, conc_tmp_9220)) {\n        float v1_9939 = ((__global float *) A_expanded_mem_10056)[gtid_9606 *\n                                                                  (conc_tmp_9220 *\n                                                                   sizze_9184) +\n                                                                  0 *\n                                                                  conc_tmp_9220 +\n                                                                  i_9862];\n        bool index_primexp_9996 = slt32(gtid_9607, y_9236);\n        int32_t index_primexp_9995 = 1 + gtid_9607;\n        float x_9944 = ((__global float *) A_expanded_mem_10056)[gtid_9606 *\n                                                                 (conc_tmp_9220 *\n                                                                  sizze_9184) +\n                                                                 0 *\n                                                                 conc_tmp_9220 +\n                                                                 gtid_9608];\n        float x_9945 = x_9944 / v1_9939;\n        float res_9946;\n        \n        if (index_primexp_9996) {\n            float x_9942 = ((__global float *) mem_10070)[gtid_9606 *\n                                                          sizze_9184 +\n                                                          gtid_9607];\n            float x_9947 = ((__global float *) A_expanded_mem_10056)[gtid_9606 *\n                                                                     (conc_tmp_9220 *\n   ",
                   "                                                                   sizze_9184) +\n                                                                     index_primexp_9995 *\n                                                                     conc_tmp_9220 +\n                                                                     gtid_9608];\n            float y_9948 = x_9942 * x_9945;\n            float res_9949 = x_9947 - y_9948;\n            \n            res_9946 = res_9949;\n        } else {\n            res_9946 = x_9945;\n        }\n        ((__global float *) mem_10077)[gtid_9606 * (conc_tmp_9220 *\n                                                    sizze_9184) + gtid_9607 *\n                                       conc_tmp_9220 + gtid_9608] = res_9946;\n    }\n}\n__kernel void segmap_9664(int32_t sizze_9183, int32_t sizze_9184,\n                          int32_t conc_tmp_9220, int32_t y_9236, int32_t i_9862,\n                          __global unsigned char *A_expanded_mem_10056, __global\n                          unsigned char *mem_10060, __global\n                          unsigned char *mem_10065, __global\n                          unsigned char *mem_10070)\n{\n    const int32_t segmap_group_sizze_9891 = mainzisegmap_group_sizze_9669;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_10161;\n    int32_t local_tid_10162;\n    int32_t group_sizze_10165;\n    int32_t wave_sizze_10164;\n    int32_t group_tid_10163;\n    \n    global_tid_10161 = get_global_id(0);\n    local_tid_10162 = get_local_id(0);\n    group_sizze_10165 = get_local_size(0);\n    wave_sizze_10164 = LOCKSTEP_WIDTH;\n    group_tid_10163 = get_group_id(0);\n    \n    int32_t phys_tid_9664 = global_tid_10161;\n    int32_t gtid_9662 = squot32(group_tid_10163 * segmap_group_sizze_9891 +\n                                local_tid_10162, sizze_9184);\n    int32_t gtid_9663;\n    \n    gtid_9663 = group_tid_10163 * segmap_group_sizze_9891 + local_tid_10162 -\n        squot32(gr",
                   "oup_tid_10163 * segmap_group_sizze_9891 + local_tid_10162,\n                sizze_9184) * sizze_9184;\n    if (slt32(gtid_9662, sizze_9183) && slt32(gtid_9663, sizze_9184)) {\n        bool cond_9903 = slt32(gtid_9663, y_9236);\n        int32_t i_9904 = 1 + gtid_9663;\n        float x_9905;\n        \n        if (cond_9903) {\n            float x_9906 = ((__global float *) A_expanded_mem_10056)[gtid_9662 *\n                                                                     (conc_tmp_9220 *\n                                                                      sizze_9184) +\n                                                                     i_9904 *\n                                                                     conc_tmp_9220 +\n                                                                     i_9862];\n            \n            x_9905 = x_9906;\n        } else {\n            x_9905 = 0.0F;\n        }\n        ((__global bool *) mem_10060)[gtid_9662 * sizze_9184 + gtid_9663] =\n            cond_9903;\n        ((__global int32_t *) mem_10065)[gtid_9662 * sizze_9184 + gtid_9663] =\n            i_9904;\n        ((__global float *) mem_10070)[gtid_9662 * sizze_9184 + gtid_9663] =\n            x_9905;\n    }\n}\n__kernel void segmap_intragroup_9377(__local volatile\n                                     int64_t *mem_10019_backing_aligned_0,\n                                     __local volatile\n                                     int64_t *mem_10022_backing_aligned_1,\n                                     __local volatile\n                                     int64_t *mem_10025_backing_aligned_2,\n                                     __local volatile\n                                     int64_t *mem_10028_backing_aligned_3,\n                                     __local volatile\n                                     int64_t *double_buffer_mem_10095_backing_aligned_4,\n                                     __local volatile\n                                     int64_t *mem_10033_backing_aligned_5,\n ",
                   "                                    int32_t sizze_9183, int32_t sizze_9184,\n                                     int32_t sizze_9185, int32_t conc_tmp_9220,\n                                     int32_t y_9236,\n                                     int32_t computed_group_sizze_9350, __global\n                                     unsigned char *mem_10009, __global\n                                     unsigned char *mem_10014, __global\n                                     unsigned char *mem_10041)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict mem_10019_backing_0 = (__local volatile\n                                                           char *) mem_10019_backing_aligned_0;\n    __local volatile char *restrict mem_10022_backing_1 = (__local volatile\n                                                           char *) mem_10022_backing_aligned_1;\n    __local volatile char *restrict mem_10025_backing_2 = (__local volatile\n                                                           char *) mem_10025_backing_aligned_2;\n    __local volatile char *restrict mem_10028_backing_3 = (__local volatile\n                                                           char *) mem_10028_backing_aligned_3;\n    __local volatile char *restrict double_buffer_mem_10095_backing_4 =\n                          (__local volatile\n                           char *) double_buffer_mem_10095_backing_aligned_4;\n    __local volatile char *restrict mem_10033_backing_5 = (__local volatile\n                                                           char *) mem_10033_backing_aligned_5;\n    int32_t global_tid_10129;\n    int32_t local_tid_10130;\n    int32_t group_sizze_10133;\n    int32_t wave_sizze_10132;\n    int32_t group_tid_10131;\n    \n    global_tid_10129 = get_global_id(0);\n    local_tid_10130 = get_local_id(0);\n    group_sizze_10133 = get_local_size(0);\n    wave_sizze_10132 = LOCKSTEP_WIDTH;\n    group_tid_10131 = get_group_id(0);\n  ",
                   "  \n    int32_t phys_tid_9377 = group_tid_10131;\n    int32_t gtid_9348 = group_tid_10131;\n    __local char *mem_10019;\n    \n    mem_10019 = (__local char *) mem_10019_backing_0;\n    \n    int32_t tmp_offs_10134 = 0;\n    \n    for (int32_t i_10135 = 0; i_10135 < sizze_9184; i_10135++) {\n        for (int32_t i_10136 = 0; i_10136 < sizze_9184; i_10136++) {\n            ((__local float *) mem_10019)[tmp_offs_10134 + (i_10135 *\n                                                            conc_tmp_9220 +\n                                                            i_10136)] =\n                ((__global float *) mem_10014)[gtid_9348 + (i_10135 *\n                                                            (sizze_9183 *\n                                                             sizze_9185) +\n                                                            i_10136 *\n                                                            sizze_9183)];\n        }\n    }\n    tmp_offs_10134 += sizze_9184;\n    for (int32_t i_10137 = 0; i_10137 < sizze_9184; i_10137++) {\n        for (int32_t i_10138 = 0; i_10138 < sizze_9184; i_10138++) {\n            ((__local float *) mem_10019)[tmp_offs_10134 + (i_10137 *\n                                                            conc_tmp_9220 +\n                                                            i_10138)] =\n                ((__global float *) mem_10009)[i_10137 * sizze_9184 + i_10138];\n        }\n    }\n    tmp_offs_10134 += sizze_9184;\n    \n    __local char *mem_10022;\n    \n    mem_10022 = (__local char *) mem_10022_backing_1;\n    \n    __local char *mem_10025;\n    \n    mem_10025 = (__local char *) mem_10025_backing_2;\n    \n    __local char *mem_10028;\n    \n    mem_10028 = (__local char *) mem_10028_backing_3;\n    \n    __local char *double_buffer_mem_10095;\n    \n    double_buffer_mem_10095 = (__local\n                               char *) double_buffer_mem_10095_backing_4;\n    for (int32_t i_10139 = 0; i_10139 < squot32(sizze_9184 * conc_tmp_9220 -\n            ",
                   "                                    local_tid_10130 +\n                                                computed_group_sizze_9350 - 1,\n                                                computed_group_sizze_9350);\n         i_10139++) {\n        ((__local float *) double_buffer_mem_10095)[squot32(i_10139 *\n                                                            computed_group_sizze_9350 +\n                                                            local_tid_10130,\n                                                            conc_tmp_9220) *\n                                                    conc_tmp_9220 + (i_10139 *\n                                                                     computed_group_sizze_9350 +\n                                                                     local_tid_10130 -\n                                                                     squot32(i_10139 *\n                                                                             computed_group_sizze_9350 +\n                                                                             local_tid_10130,\n                                                                             conc_tmp_9220) *\n                                                                     conc_tmp_9220)] =\n            ((__local float *) mem_10019)[squot32(i_10139 *\n                                                  computed_group_sizze_9350 +\n                                                  local_tid_10130,\n                                                  conc_tmp_9220) *\n                                          conc_tmp_9220 + (i_10139 *\n                                                           computed_group_sizze_9350 +\n                                                           local_tid_10130 -\n                                                           squot32(i_10139 *\n                                                                   computed_group_sizze_9350 +\n                                                  ",
                   "                 local_tid_10130,\n                                                                   conc_tmp_9220) *\n                                                           conc_tmp_9220)];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    __local char *mem_10033;\n    \n    mem_10033 = (__local char *) mem_10033_backing_5;\n    for (int32_t i_9448 = 0; i_9448 < sizze_9184; i_9448++) {\n        float v1_9449 = ((__local float *) double_buffer_mem_10095)[0 *\n                                                                    conc_tmp_9220 +\n                                                                    i_9448];\n        int32_t gtid_9367 = local_tid_10130;\n        int32_t phys_tid_9368;\n        \n        phys_tid_9368 = local_tid_10130;\n        if (slt32(gtid_9367, sizze_9184)) {\n            bool cond_9454 = slt32(gtid_9367, y_9236);\n            int32_t i_9455 = 1 + gtid_9367;\n            float x_9456;\n            \n            if (cond_9454) {\n                float x_9457 = ((__local\n                                 float *) double_buffer_mem_10095)[i_9455 *\n                                                                   conc_tmp_9220 +\n                                                                   i_9448];\n                \n                x_9456 = x_9457;\n            } else {\n                x_9456 = 0.0F;\n            }\n            ((__local bool *) mem_10022)[gtid_9367] = cond_9454;\n            ((__local int32_t *) mem_10025)[gtid_9367] = i_9455;\n            ((__local float *) mem_10028)[gtid_9367] = x_9456;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        \n        int32_t gtid_9354 = squot32(local_tid_10130, conc_tmp_9220);\n        int32_t gtid_9355;\n        \n        gtid_9355 = local_tid_10130 - squot32(local_tid_10130, conc_tmp_9220) *\n            conc_tmp_9220;\n        \n        int32_t phys_tid_9356;\n        \n        phys_tid_9356 = local_tid_10130;\n        if (slt32(gtid_9354, sizze_9184) && slt32(gtid_9355, conc_tmp_9220)) {\n            bool i",
                   "ndex_primexp_9994 = slt32(gtid_9354, y_9236);\n            int32_t index_primexp_9993 = 1 + gtid_9354;\n            float x_9463 = ((__local float *) double_buffer_mem_10095)[0 *\n                                                                       conc_tmp_9220 +\n                                                                       gtid_9355];\n            float x_9464 = x_9463 / v1_9449;\n            float res_9465;\n            \n            if (index_primexp_9994) {\n                float x_9461 = ((__local float *) mem_10028)[gtid_9354];\n                float x_9466 = ((__local\n                                 float *) double_buffer_mem_10095)[index_primexp_9993 *\n                                                                   conc_tmp_9220 +\n                                                                   gtid_9355];\n                float y_9467 = x_9461 * x_9464;\n                float res_9468 = x_9466 - y_9467;\n                \n                res_9465 = res_9468;\n            } else {\n                res_9465 = x_9464;\n            }\n            ((__local float *) mem_10033)[gtid_9354 * conc_tmp_9220 +\n                                          gtid_9355] = res_9465;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int32_t i_10141 = 0; i_10141 < squot32(sizze_9184 * conc_tmp_9220 -\n                                                    local_tid_10130 +\n                                                    computed_group_sizze_9350 -\n                                                    1,\n                                                    computed_group_sizze_9350);\n             i_10141++) {\n            ((__local float *) double_buffer_mem_10095)[squot32(i_10141 *\n                                                                computed_group_sizze_9350 +\n                                                                local_tid_10130,\n                                                                conc_tmp_9220) *\n                                        ",
                   "                conc_tmp_9220 +\n                                                        (i_10141 *\n                                                         computed_group_sizze_9350 +\n                                                         local_tid_10130 -\n                                                         squot32(i_10141 *\n                                                                 computed_group_sizze_9350 +\n                                                                 local_tid_10130,\n                                                                 conc_tmp_9220) *\n                                                         conc_tmp_9220)] =\n                ((__local float *) mem_10033)[squot32(i_10141 *\n                                                      computed_group_sizze_9350 +\n                                                      local_tid_10130,\n                                                      conc_tmp_9220) *\n                                              conc_tmp_9220 + (i_10141 *\n                                                               computed_group_sizze_9350 +\n                                                               local_tid_10130 -\n                                                               squot32(i_10141 *\n                                                                       computed_group_sizze_9350 +\n                                                                       local_tid_10130,\n                                                                       conc_tmp_9220) *\n                                                               conc_tmp_9220)];\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    for (int32_t i_10142 = 0; i_10142 < squot32(sizze_9184 * sizze_9184 -\n                                                local_tid_10130 +\n                                                computed_group_sizze_9350 - 1,\n                                                computed_group_sizze_9350);\n         i_10142",
                   "++) {\n        ((__global float *) mem_10041)[gtid_9348 * (sizze_9184 * sizze_9184) +\n                                       squot32(i_10142 *\n                                               computed_group_sizze_9350 +\n                                               local_tid_10130, sizze_9184) *\n                                       sizze_9184 + (i_10142 *\n                                                     computed_group_sizze_9350 +\n                                                     local_tid_10130 -\n                                                     squot32(i_10142 *\n                                                             computed_group_sizze_9350 +\n                                                             local_tid_10130,\n                                                             sizze_9184) *\n                                                     sizze_9184)] = ((__local\n                                                                      float *) double_buffer_mem_10095)[sizze_9184 +\n                                                                                                        (squot32(i_10142 *\n                                                                                                                 computed_group_sizze_9350 +\n                                                                                                                 local_tid_10130,\n                                                                                                                 sizze_9184) *\n                                                                                                         conc_tmp_9220 +\n                                                                                                         (i_10142 *\n                                                                                                          computed_group_sizze_9350 +\n                                                                                          ",
                   "                local_tid_10130 -\n                                                                                                          squot32(i_10142 *\n                                                                                                                  computed_group_sizze_9350 +\n                                                                                                                  local_tid_10130,\n                                                                                                                  sizze_9184) *\n                                                                                                          sizze_9184))];\n    }\n}\n",
                   NULL};
struct memblock_device {
    int *references;
    cl_mem mem;
    int64_t size;
    const char *desc;
} ;
struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
static const char *size_names[] = {"main.group_size_10122",
                                   "main.group_size_10146",
                                   "main.group_size_10152",
                                   "main.group_size_10157",
                                   "main.group_size_10174",
                                   "main.segmap_group_size_9307",
                                   "main.segmap_group_size_9616",
                                   "main.segmap_group_size_9669",
                                   "main.suff_intra_par_2"};
static const char *size_vars[] = {"mainzigroup_sizze_10122",
                                  "mainzigroup_sizze_10146",
                                  "mainzigroup_sizze_10152",
                                  "mainzigroup_sizze_10157",
                                  "mainzigroup_sizze_10174",
                                  "mainzisegmap_group_sizze_9307",
                                  "mainzisegmap_group_sizze_9616",
                                  "mainzisegmap_group_sizze_9669",
                                  "mainzisuff_intra_par_2"};
static const char *size_classes[] = {"group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size",
                                     "threshold ()"};
int futhark_get_num_sizes(void)
{
    return 9;
}
const char *futhark_get_size_name(int i)
{
    return size_names[i];
}
const char *futhark_get_size_class(int i)
{
    return size_classes[i];
}
struct sizes {
    size_t mainzigroup_sizze_10122;
    size_t mainzigroup_sizze_10146;
    size_t mainzigroup_sizze_10152;
    size_t mainzigroup_sizze_10157;
    size_t mainzigroup_sizze_10174;
    size_t mainzisegmap_group_sizze_9307;
    size_t mainzisegmap_group_sizze_9616;
    size_t mainzisegmap_group_sizze_9669;
    size_t mainzisuff_intra_par_2;
} ;
struct futhark_context_config {
    struct opencl_config opencl;
    size_t sizes[9];
    int num_build_opts;
    const char **build_opts;
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  (struct futhark_context_config *) malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->num_build_opts = 0;
    cfg->build_opts = (const char **) malloc(sizeof(const char *));
    cfg->build_opts[0] = NULL;
    cfg->sizes[0] = 0;
    cfg->sizes[1] = 0;
    cfg->sizes[2] = 0;
    cfg->sizes[3] = 0;
    cfg->sizes[4] = 0;
    cfg->sizes[5] = 0;
    cfg->sizes[6] = 0;
    cfg->sizes[7] = 0;
    cfg->sizes[8] = 0;
    opencl_config_init(&cfg->opencl, 9, size_names, size_vars, cfg->sizes,
                       size_classes);
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg->build_opts);
    free(cfg);
}
void futhark_context_config_add_build_option(struct futhark_context_config *cfg,
                                             const char *opt)
{
    cfg->build_opts[cfg->num_build_opts] = opt;
    cfg->num_build_opts++;
    cfg->build_opts = (const char **) realloc(cfg->build_opts,
                                              (cfg->num_build_opts + 1) *
                                              sizeof(const char *));
    cfg->build_opts[cfg->num_build_opts] = NULL;
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->opencl.profiling = cfg->opencl.logging = cfg->opencl.debugging = flag;
}
void futhark_context_config_set_profiling(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->opencl.profiling = flag;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag)
{
    cfg->opencl.logging = flag;
}
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s)
{
    set_preferred_device(&cfg->opencl, s);
}
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s)
{
    set_preferred_platform(&cfg->opencl, s);
}
void futhark_context_config_select_device_interactively(struct futhark_context_config *cfg)
{
    select_device_interactively(&cfg->opencl);
}
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path)
{
    cfg->opencl.dump_program_to = path;
}
void futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                              const char *path)
{
    cfg->opencl.load_program_from = path;
}
void futhark_context_config_dump_binary_to(struct futhark_context_config *cfg,
                                           const char *path)
{
    cfg->opencl.dump_binary_to = path;
}
void futhark_context_config_load_binary_from(struct futhark_context_config *cfg,
                                             const char *path)
{
    cfg->opencl.load_binary_from = path;
}
void futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                                   int size)
{
    cfg->opencl.default_group_size = size;
    cfg->opencl.default_group_size_changed = 1;
}
void futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                                   int num)
{
    cfg->opencl.default_num_groups = num;
}
void futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_tile_size = size;
    cfg->opencl.default_tile_size_changed = 1;
}
void futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_threshold = size;
}
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value)
{
    for (int i = 0; i < 9; i++) {
        if (strcmp(size_name, size_names[i]) == 0) {
            cfg->sizes[i] = size_value;
            return 0;
        }
    }
    if (strcmp(size_name, "default_group_size") == 0) {
        cfg->opencl.default_group_size = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_num_groups") == 0) {
        cfg->opencl.default_num_groups = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_threshold") == 0) {
        cfg->opencl.default_threshold = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_tile_size") == 0) {
        cfg->opencl.default_tile_size = size_value;
        return 0;
    }
    return 1;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    int profiling;
    int profiling_paused;
    int logging;
    lock_t lock;
    char *error;
    int64_t peak_mem_usage_device;
    int64_t cur_mem_usage_device;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
    int total_runs;
    long total_runtime;
    cl_kernel copy_10149;
    cl_kernel copy_10154;
    cl_kernel copy_10171;
    cl_kernel map_transpose_f32;
    cl_kernel map_transpose_f32_low_height;
    cl_kernel map_transpose_f32_low_width;
    cl_kernel map_transpose_f32_small;
    cl_kernel replicate_10119;
    cl_kernel replicate_10143;
    cl_kernel segmap_9302;
    cl_kernel segmap_9609;
    cl_kernel segmap_9664;
    cl_kernel segmap_intragroup_9377;
    int64_t copy_dev_to_dev_total_runtime;
    int copy_dev_to_dev_runs;
    int64_t copy_dev_to_host_total_runtime;
    int copy_dev_to_host_runs;
    int64_t copy_host_to_dev_total_runtime;
    int copy_host_to_dev_runs;
    int64_t copy_scalar_to_dev_total_runtime;
    int copy_scalar_to_dev_runs;
    int64_t copy_scalar_from_dev_total_runtime;
    int copy_scalar_from_dev_runs;
    int64_t copy_10149_total_runtime;
    int copy_10149_runs;
    int64_t copy_10154_total_runtime;
    int copy_10154_runs;
    int64_t copy_10171_total_runtime;
    int copy_10171_runs;
    int64_t map_transpose_f32_total_runtime;
    int map_transpose_f32_runs;
    int64_t map_transpose_f32_low_height_total_runtime;
    int map_transpose_f32_low_height_runs;
    int64_t map_transpose_f32_low_width_total_runtime;
    int map_transpose_f32_low_width_runs;
    int64_t map_transpose_f32_small_total_runtime;
    int map_transpose_f32_small_runs;
    int64_t replicate_10119_total_runtime;
    int replicate_10119_runs;
    int64_t replicate_10143_total_runtime;
    int replicate_10143_runs;
    int64_t segmap_9302_total_runtime;
    int segmap_9302_runs;
    int64_t segmap_9609_total_runtime;
    int segmap_9609_runs;
    int64_t segmap_9664_total_runtime;
    int segmap_9664_runs;
    int64_t segmap_intragroup_9377_total_runtime;
    int segmap_intragroup_9377_runs;
    struct opencl_context opencl;
    struct sizes sizes;
} ;
void post_opencl_setup(struct opencl_context *ctx,
                       struct opencl_device_option *option)
{
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "NVIDIA CUDA") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 32;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "AMD Accelerated Parallel Processing") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 32;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 1;
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_num_groups = 256;
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_GPU) == CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_group_size = 256;
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_tile_size = 32;
    if ((ctx->cfg.default_threshold == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_GPU) ==
        CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_threshold = 32768;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU)
        ctx->lockstep_width = 1;
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU)
        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(ctx->cfg.default_num_groups),
                        &ctx->cfg.default_num_groups, NULL);
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        (option->device_type & CL_DEVICE_TYPE_CPU) == CL_DEVICE_TYPE_CPU)
        ctx->cfg.default_group_size = 32;
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU)
        ctx->cfg.default_tile_size = 4;
    if ((ctx->cfg.default_threshold == 0 && strstr(option->platform_name, "") !=
         NULL) && (option->device_type & CL_DEVICE_TYPE_CPU) ==
        CL_DEVICE_TYPE_CPU)
        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(ctx->cfg.default_threshold),
                        &ctx->cfg.default_threshold, NULL);
}
static void init_context_early(struct futhark_context_config *cfg,
                               struct futhark_context *ctx)
{
    ctx->opencl.cfg = cfg->opencl;
    ctx->detail_memory = cfg->opencl.debugging;
    ctx->debugging = cfg->opencl.debugging;
    ctx->profiling = cfg->opencl.profiling;
    ctx->profiling_paused = 0;
    ctx->logging = cfg->opencl.logging;
    ctx->error = NULL;
    ctx->opencl.profiling_records_capacity = 200;
    ctx->opencl.profiling_records_used = 0;
    ctx->opencl.profiling_records =
        malloc(ctx->opencl.profiling_records_capacity *
        sizeof(struct profiling_record));
    create_lock(&ctx->lock);
    ctx->peak_mem_usage_device = 0;
    ctx->cur_mem_usage_device = 0;
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    ctx->total_runs = 0;
    ctx->total_runtime = 0;
    ctx->copy_dev_to_dev_total_runtime = 0;
    ctx->copy_dev_to_dev_runs = 0;
    ctx->copy_dev_to_host_total_runtime = 0;
    ctx->copy_dev_to_host_runs = 0;
    ctx->copy_host_to_dev_total_runtime = 0;
    ctx->copy_host_to_dev_runs = 0;
    ctx->copy_scalar_to_dev_total_runtime = 0;
    ctx->copy_scalar_to_dev_runs = 0;
    ctx->copy_scalar_from_dev_total_runtime = 0;
    ctx->copy_scalar_from_dev_runs = 0;
    ctx->copy_10149_total_runtime = 0;
    ctx->copy_10149_runs = 0;
    ctx->copy_10154_total_runtime = 0;
    ctx->copy_10154_runs = 0;
    ctx->copy_10171_total_runtime = 0;
    ctx->copy_10171_runs = 0;
    ctx->map_transpose_f32_total_runtime = 0;
    ctx->map_transpose_f32_runs = 0;
    ctx->map_transpose_f32_low_height_total_runtime = 0;
    ctx->map_transpose_f32_low_height_runs = 0;
    ctx->map_transpose_f32_low_width_total_runtime = 0;
    ctx->map_transpose_f32_low_width_runs = 0;
    ctx->map_transpose_f32_small_total_runtime = 0;
    ctx->map_transpose_f32_small_runs = 0;
    ctx->replicate_10119_total_runtime = 0;
    ctx->replicate_10119_runs = 0;
    ctx->replicate_10143_total_runtime = 0;
    ctx->replicate_10143_runs = 0;
    ctx->segmap_9302_total_runtime = 0;
    ctx->segmap_9302_runs = 0;
    ctx->segmap_9609_total_runtime = 0;
    ctx->segmap_9609_runs = 0;
    ctx->segmap_9664_total_runtime = 0;
    ctx->segmap_9664_runs = 0;
    ctx->segmap_intragroup_9377_total_runtime = 0;
    ctx->segmap_intragroup_9377_runs = 0;
}
static int init_context_late(struct futhark_context_config *cfg,
                             struct futhark_context *ctx, cl_program prog)
{
    cl_int error;
    
    {
        ctx->copy_10149 = clCreateKernel(prog, "copy_10149", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "copy_10149");
    }
    {
        ctx->copy_10154 = clCreateKernel(prog, "copy_10154", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "copy_10154");
    }
    {
        ctx->copy_10171 = clCreateKernel(prog, "copy_10171", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "copy_10171");
    }
    {
        ctx->map_transpose_f32 = clCreateKernel(prog, "map_transpose_f32",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_transpose_f32");
    }
    {
        ctx->map_transpose_f32_low_height = clCreateKernel(prog,
                                                           "map_transpose_f32_low_height",
                                                           &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "map_transpose_f32_low_height");
    }
    {
        ctx->map_transpose_f32_low_width = clCreateKernel(prog,
                                                          "map_transpose_f32_low_width",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "map_transpose_f32_low_width");
    }
    {
        ctx->map_transpose_f32_small = clCreateKernel(prog,
                                                      "map_transpose_f32_small",
                                                      &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_transpose_f32_small");
    }
    {
        ctx->replicate_10119 = clCreateKernel(prog, "replicate_10119", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "replicate_10119");
    }
    {
        ctx->replicate_10143 = clCreateKernel(prog, "replicate_10143", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "replicate_10143");
    }
    {
        ctx->segmap_9302 = clCreateKernel(prog, "segmap_9302", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_9302");
    }
    {
        ctx->segmap_9609 = clCreateKernel(prog, "segmap_9609", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_9609");
    }
    {
        ctx->segmap_9664 = clCreateKernel(prog, "segmap_9664", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_9664");
    }
    {
        ctx->segmap_intragroup_9377 = clCreateKernel(prog,
                                                     "segmap_intragroup_9377",
                                                     &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_intragroup_9377");
    }
    ctx->sizes.mainzigroup_sizze_10122 = cfg->sizes[0];
    ctx->sizes.mainzigroup_sizze_10146 = cfg->sizes[1];
    ctx->sizes.mainzigroup_sizze_10152 = cfg->sizes[2];
    ctx->sizes.mainzigroup_sizze_10157 = cfg->sizes[3];
    ctx->sizes.mainzigroup_sizze_10174 = cfg->sizes[4];
    ctx->sizes.mainzisegmap_group_sizze_9307 = cfg->sizes[5];
    ctx->sizes.mainzisegmap_group_sizze_9616 = cfg->sizes[6];
    ctx->sizes.mainzisegmap_group_sizze_9669 = cfg->sizes[7];
    ctx->sizes.mainzisuff_intra_par_2 = cfg->sizes[8];
    return 0;
}
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    
    int required_types = 0;
    
    init_context_early(cfg, ctx);
    
    cl_program prog = setup_opencl(&ctx->opencl, opencl_program, required_types,
                                   cfg->build_opts);
    
    init_context_late(cfg, ctx, prog);
    return ctx;
}
struct futhark_context *futhark_context_new_with_command_queue(struct futhark_context_config *cfg,
                                                               cl_command_queue queue)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    
    int required_types = 0;
    
    init_context_early(cfg, ctx);
    
    cl_program prog = setup_opencl_with_command_queue(&ctx->opencl, queue,
                                                      opencl_program,
                                                      required_types,
                                                      cfg->build_opts);
    
    init_context_late(cfg, ctx, prog);
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_lock(&ctx->lock);
    opencl_tally_profiling_records(&ctx->opencl);
    free(ctx->opencl.profiling_records);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    ctx->error = OPENCL_SUCCEED_NONFATAL(clFinish(ctx->opencl.queue));
    return ctx->error != NULL;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
void futhark_context_pause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 1;
}
void futhark_context_unpause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 0;
}
int futhark_context_clear_caches(struct futhark_context *ctx)
{
    ctx->error = OPENCL_SUCCEED_NONFATAL(opencl_free_all(&ctx->opencl));
    return ctx->error != NULL;
}
cl_command_queue futhark_context_get_command_queue(struct futhark_context *ctx)
{
    return ctx->opencl.queue;
}
static int memblock_unref_device(struct futhark_context *ctx,
                                 struct memblock_device *block, const
                                 char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'device'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_device -= block->size;
            OPENCL_SUCCEED_OR_RETURN(opencl_free(&ctx->opencl, block->mem,
                                                 block->desc));
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_device);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc_device(struct futhark_context *ctx,
                                 struct memblock_device *block, int64_t size,
                                 const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "space 'device'",
              ctx->cur_mem_usage_device);
    
    int ret = memblock_unref_device(ctx, block, desc);
    
    ctx->cur_mem_usage_device += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "space 'device'",
                (long long) ctx->cur_mem_usage_device);
    if (ctx->cur_mem_usage_device > ctx->peak_mem_usage_device) {
        ctx->peak_mem_usage_device = ctx->cur_mem_usage_device;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    OPENCL_SUCCEED_OR_RETURN(opencl_alloc(&ctx->opencl, size, desc,
                                          &block->mem));
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set_device(struct futhark_context *ctx,
                               struct memblock_device *lhs,
                               struct memblock_device *rhs, const
                               char *lhs_desc)
{
    int ret = memblock_unref_device(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "default space",
              ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
void futhark_debugging_report(struct futhark_context *ctx)
{
    if (ctx->detail_memory || ctx->profiling) {
        fprintf(stderr, "Peak memory usage for space 'device': %lld bytes.\n",
                (long long) ctx->peak_mem_usage_device);
        fprintf(stderr, "Peak memory usage for default space: %lld bytes.\n",
                (long long) ctx->peak_mem_usage_default);
    }
    if (ctx->profiling) {
        OPENCL_SUCCEED_FATAL(opencl_tally_profiling_records(&ctx->opencl));
        fprintf(stderr,
                "copy_dev_to_dev              ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->copy_dev_to_dev_runs,
                (long) ctx->copy_dev_to_dev_total_runtime /
                (ctx->copy_dev_to_dev_runs !=
                 0 ? ctx->copy_dev_to_dev_runs : 1),
                (long) ctx->copy_dev_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_dev_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_dev_to_dev_runs;
        fprintf(stderr,
                "copy_dev_to_host             ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->copy_dev_to_host_runs,
                (long) ctx->copy_dev_to_host_total_runtime /
                (ctx->copy_dev_to_host_runs !=
                 0 ? ctx->copy_dev_to_host_runs : 1),
                (long) ctx->copy_dev_to_host_total_runtime);
        ctx->total_runtime += ctx->copy_dev_to_host_total_runtime;
        ctx->total_runs += ctx->copy_dev_to_host_runs;
        fprintf(stderr,
                "copy_host_to_dev             ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->copy_host_to_dev_runs,
                (long) ctx->copy_host_to_dev_total_runtime /
                (ctx->copy_host_to_dev_runs !=
                 0 ? ctx->copy_host_to_dev_runs : 1),
                (long) ctx->copy_host_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_host_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_host_to_dev_runs;
        fprintf(stderr,
                "copy_scalar_to_dev           ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->copy_scalar_to_dev_runs,
                (long) ctx->copy_scalar_to_dev_total_runtime /
                (ctx->copy_scalar_to_dev_runs !=
                 0 ? ctx->copy_scalar_to_dev_runs : 1),
                (long) ctx->copy_scalar_to_dev_total_runtime);
        ctx->total_runtime += ctx->copy_scalar_to_dev_total_runtime;
        ctx->total_runs += ctx->copy_scalar_to_dev_runs;
        fprintf(stderr,
                "copy_scalar_from_dev         ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->copy_scalar_from_dev_runs,
                (long) ctx->copy_scalar_from_dev_total_runtime /
                (ctx->copy_scalar_from_dev_runs !=
                 0 ? ctx->copy_scalar_from_dev_runs : 1),
                (long) ctx->copy_scalar_from_dev_total_runtime);
        ctx->total_runtime += ctx->copy_scalar_from_dev_total_runtime;
        ctx->total_runs += ctx->copy_scalar_from_dev_runs;
        fprintf(stderr,
                "copy_10149                   ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->copy_10149_runs, (long) ctx->copy_10149_total_runtime /
                (ctx->copy_10149_runs != 0 ? ctx->copy_10149_runs : 1),
                (long) ctx->copy_10149_total_runtime);
        ctx->total_runtime += ctx->copy_10149_total_runtime;
        ctx->total_runs += ctx->copy_10149_runs;
        fprintf(stderr,
                "copy_10154                   ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->copy_10154_runs, (long) ctx->copy_10154_total_runtime /
                (ctx->copy_10154_runs != 0 ? ctx->copy_10154_runs : 1),
                (long) ctx->copy_10154_total_runtime);
        ctx->total_runtime += ctx->copy_10154_total_runtime;
        ctx->total_runs += ctx->copy_10154_runs;
        fprintf(stderr,
                "copy_10171                   ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->copy_10171_runs, (long) ctx->copy_10171_total_runtime /
                (ctx->copy_10171_runs != 0 ? ctx->copy_10171_runs : 1),
                (long) ctx->copy_10171_total_runtime);
        ctx->total_runtime += ctx->copy_10171_total_runtime;
        ctx->total_runs += ctx->copy_10171_runs;
        fprintf(stderr,
                "map_transpose_f32            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->map_transpose_f32_runs,
                (long) ctx->map_transpose_f32_total_runtime /
                (ctx->map_transpose_f32_runs !=
                 0 ? ctx->map_transpose_f32_runs : 1),
                (long) ctx->map_transpose_f32_total_runtime);
        ctx->total_runtime += ctx->map_transpose_f32_total_runtime;
        ctx->total_runs += ctx->map_transpose_f32_runs;
        fprintf(stderr,
                "map_transpose_f32_low_height ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->map_transpose_f32_low_height_runs,
                (long) ctx->map_transpose_f32_low_height_total_runtime /
                (ctx->map_transpose_f32_low_height_runs !=
                 0 ? ctx->map_transpose_f32_low_height_runs : 1),
                (long) ctx->map_transpose_f32_low_height_total_runtime);
        ctx->total_runtime += ctx->map_transpose_f32_low_height_total_runtime;
        ctx->total_runs += ctx->map_transpose_f32_low_height_runs;
        fprintf(stderr,
                "map_transpose_f32_low_width  ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->map_transpose_f32_low_width_runs,
                (long) ctx->map_transpose_f32_low_width_total_runtime /
                (ctx->map_transpose_f32_low_width_runs !=
                 0 ? ctx->map_transpose_f32_low_width_runs : 1),
                (long) ctx->map_transpose_f32_low_width_total_runtime);
        ctx->total_runtime += ctx->map_transpose_f32_low_width_total_runtime;
        ctx->total_runs += ctx->map_transpose_f32_low_width_runs;
        fprintf(stderr,
                "map_transpose_f32_small      ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->map_transpose_f32_small_runs,
                (long) ctx->map_transpose_f32_small_total_runtime /
                (ctx->map_transpose_f32_small_runs !=
                 0 ? ctx->map_transpose_f32_small_runs : 1),
                (long) ctx->map_transpose_f32_small_total_runtime);
        ctx->total_runtime += ctx->map_transpose_f32_small_total_runtime;
        ctx->total_runs += ctx->map_transpose_f32_small_runs;
        fprintf(stderr,
                "replicate_10119              ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->replicate_10119_runs,
                (long) ctx->replicate_10119_total_runtime /
                (ctx->replicate_10119_runs !=
                 0 ? ctx->replicate_10119_runs : 1),
                (long) ctx->replicate_10119_total_runtime);
        ctx->total_runtime += ctx->replicate_10119_total_runtime;
        ctx->total_runs += ctx->replicate_10119_runs;
        fprintf(stderr,
                "replicate_10143              ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->replicate_10143_runs,
                (long) ctx->replicate_10143_total_runtime /
                (ctx->replicate_10143_runs !=
                 0 ? ctx->replicate_10143_runs : 1),
                (long) ctx->replicate_10143_total_runtime);
        ctx->total_runtime += ctx->replicate_10143_total_runtime;
        ctx->total_runs += ctx->replicate_10143_runs;
        fprintf(stderr,
                "segmap_9302                  ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_9302_runs, (long) ctx->segmap_9302_total_runtime /
                (ctx->segmap_9302_runs != 0 ? ctx->segmap_9302_runs : 1),
                (long) ctx->segmap_9302_total_runtime);
        ctx->total_runtime += ctx->segmap_9302_total_runtime;
        ctx->total_runs += ctx->segmap_9302_runs;
        fprintf(stderr,
                "segmap_9609                  ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_9609_runs, (long) ctx->segmap_9609_total_runtime /
                (ctx->segmap_9609_runs != 0 ? ctx->segmap_9609_runs : 1),
                (long) ctx->segmap_9609_total_runtime);
        ctx->total_runtime += ctx->segmap_9609_total_runtime;
        ctx->total_runs += ctx->segmap_9609_runs;
        fprintf(stderr,
                "segmap_9664                  ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_9664_runs, (long) ctx->segmap_9664_total_runtime /
                (ctx->segmap_9664_runs != 0 ? ctx->segmap_9664_runs : 1),
                (long) ctx->segmap_9664_total_runtime);
        ctx->total_runtime += ctx->segmap_9664_total_runtime;
        ctx->total_runs += ctx->segmap_9664_runs;
        fprintf(stderr,
                "segmap_intragroup_9377       ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_intragroup_9377_runs,
                (long) ctx->segmap_intragroup_9377_total_runtime /
                (ctx->segmap_intragroup_9377_runs !=
                 0 ? ctx->segmap_intragroup_9377_runs : 1),
                (long) ctx->segmap_intragroup_9377_total_runtime);
        ctx->total_runtime += ctx->segmap_intragroup_9377_total_runtime;
        ctx->total_runs += ctx->segmap_intragroup_9377_runs;
        if (ctx->profiling)
            fprintf(stderr, "%d operations with cumulative runtime: %6ldus\n",
                    ctx->total_runs, ctx->total_runtime);
    }
}
static int futrts_main(struct futhark_context *ctx,
                       struct memblock_device *out_mem_p_10176,
                       int32_t *out_out_arrsizze_10177,
                       int32_t *out_out_arrsizze_10178,
                       int32_t *out_out_arrsizze_10179,
                       struct memblock_device xs_mem_10004, int32_t sizze_9183,
                       int32_t sizze_9184, int32_t sizze_9185);
static int futrts__map_transpose_f32(struct futhark_context *ctx,
                                     struct memblock_device destmem_0,
                                     int32_t destoffset_1,
                                     struct memblock_device srcmem_2,
                                     int32_t srcoffset_3, int32_t num_arrays_4,
                                     int32_t x_elems_5, int32_t y_elems_6,
                                     int32_t in_elems_7, int32_t out_elems_8);
static int futrts__replicate_f32(struct futhark_context *ctx,
                                 struct memblock_device mem_10115,
                                 int32_t num_elems_10116, float val_10117);
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline char ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline char ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline char ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline char ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline char ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline char ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline char ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline char ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline char slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline char slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline char slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline char slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline char sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline char sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline char sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline char sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((uint8_t) (uint8_t) x)
#define zext_i8_i16(x) ((uint16_t) (uint8_t) x)
#define zext_i8_i32(x) ((uint32_t) (uint8_t) x)
#define zext_i8_i64(x) ((uint64_t) (uint8_t) x)
#define zext_i16_i8(x) ((uint8_t) (uint16_t) x)
#define zext_i16_i16(x) ((uint16_t) (uint16_t) x)
#define zext_i16_i32(x) ((uint32_t) (uint16_t) x)
#define zext_i16_i64(x) ((uint64_t) (uint16_t) x)
#define zext_i32_i8(x) ((uint8_t) (uint32_t) x)
#define zext_i32_i16(x) ((uint16_t) (uint32_t) x)
#define zext_i32_i32(x) ((uint32_t) (uint32_t) x)
#define zext_i32_i64(x) ((uint64_t) (uint32_t) x)
#define zext_i64_i8(x) ((uint8_t) (uint64_t) x)
#define zext_i64_i16(x) ((uint16_t) (uint64_t) x)
#define zext_i64_i32(x) ((uint32_t) (uint64_t) x)
#define zext_i64_i64(x) ((uint64_t) (uint64_t) x)
#ifdef __OPENCL_VERSION__
int32_t futrts_popc8(int8_t x)
{
    return popcount(x);
}
int32_t futrts_popc16(int16_t x)
{
    return popcount(x);
}
int32_t futrts_popc32(int32_t x)
{
    return popcount(x);
}
int32_t futrts_popc64(int64_t x)
{
    return popcount(x);
}
#elif __CUDA_ARCH__
int32_t futrts_popc8(int8_t x)
{
    return __popc(zext_i8_i32(x));
}
int32_t futrts_popc16(int16_t x)
{
    return __popc(zext_i16_i32(x));
}
int32_t futrts_popc32(int32_t x)
{
    return __popc(x);
}
int32_t futrts_popc64(int64_t x)
{
    return __popcll(x);
}
#else
int32_t futrts_popc8(int8_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
int32_t futrts_popc16(int16_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
int32_t futrts_popc32(int32_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
int32_t futrts_popc64(int64_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
#endif
#ifdef __OPENCL_VERSION__
int32_t futrts_clzz8(int8_t x)
{
    return clz(x);
}
int32_t futrts_clzz16(int16_t x)
{
    return clz(x);
}
int32_t futrts_clzz32(int32_t x)
{
    return clz(x);
}
int32_t futrts_clzz64(int64_t x)
{
    return clz(x);
}
#elif __CUDA_ARCH__
int32_t futrts_clzz8(int8_t x)
{
    return __clz(zext_i8_i32(x)) - 24;
}
int32_t futrts_clzz16(int16_t x)
{
    return __clz(zext_i16_i32(x)) - 16;
}
int32_t futrts_clzz32(int32_t x)
{
    return __clz(x);
}
int32_t futrts_clzz64(int64_t x)
{
    return __clzll(x);
}
#else
int32_t futrts_clzz8(int8_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
int32_t futrts_clzz16(int16_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
int32_t futrts_clzz32(int32_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
int32_t futrts_clzz64(int64_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
#endif
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return fmin(x, y);
}
static inline float fmax32(float x, float y)
{
    return fmax(x, y);
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline char cmplt32(float x, float y)
{
    return x < y;
}
static inline char cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return fmin(x, y);
}
static inline double fmax64(double x, double y)
{
    return fmax(x, y);
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline char cmplt64(double x, double y)
{
    return x < y;
}
static inline char cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return x;
}
static inline float fpconv_f32_f32(float x)
{
    return x;
}
static inline double fpconv_f32_f64(float x)
{
    return x;
}
static inline float fpconv_f64_f32(double x)
{
    return x;
}
static inline double fpconv_f64_f64(double x)
{
    return x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline char futrts_isnan32(float x)
{
    return isnan(x);
}
static inline char futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
#ifdef __OPENCL_VERSION__
static inline float fmod32(float x, float y)
{
    return fmod(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline float futrts_floor32(float x)
{
    return floor(x);
}
static inline float futrts_ceil32(float x)
{
    return ceil(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return mix(v0, v1, t);
}
#else
static inline float fmod32(float x, float y)
{
    return fmodf(x, y);
}
static inline float futrts_round32(float x)
{
    return rintf(x);
}
static inline float futrts_floor32(float x)
{
    return floorf(x);
}
static inline float futrts_ceil32(float x)
{
    return ceilf(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return v0 + (v1 - v0) * t;
}
#endif
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_gamma64(double x)
{
    return tgamma(x);
}
static inline double futrts_lgamma64(double x)
{
    return lgamma(x);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline double futrts_ceil64(double x)
{
    return ceil(x);
}
static inline double futrts_floor64(double x)
{
    return floor(x);
}
static inline char futrts_isnan64(double x)
{
    return isnan(x);
}
static inline char futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float fmod64(float x, float y)
{
    return fmod(x, y);
}
#ifdef __OPENCL_VERSION__
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return mix(v0, v1, t);
}
#else
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return v0 + (v1 - v0) * t;
}
#endif
static int futrts_main(struct futhark_context *ctx,
                       struct memblock_device *out_mem_p_10176,
                       int32_t *out_out_arrsizze_10177,
                       int32_t *out_out_arrsizze_10178,
                       int32_t *out_out_arrsizze_10179,
                       struct memblock_device xs_mem_10004, int32_t sizze_9183,
                       int32_t sizze_9184, int32_t sizze_9185)
{
    struct memblock_device out_mem_10111;
    
    out_mem_10111.references = NULL;
    
    int32_t out_arrsizze_10112;
    int32_t out_arrsizze_10113;
    int32_t out_arrsizze_10114;
    bool dim_zzero_9187 = 0 == sizze_9184;
    bool dim_zzero_9188 = 0 == sizze_9185;
    bool old_empty_9189 = dim_zzero_9187 || dim_zzero_9188;
    bool new_empty_9190 = dim_zzero_9187 || dim_zzero_9187;
    bool both_empty_9191 = old_empty_9189 && new_empty_9190;
    bool dim_match_9192 = sizze_9184 == sizze_9185;
    bool empty_or_match_9193 = both_empty_9191 || dim_match_9192;
    bool empty_or_match_cert_9194;
    
    if (!empty_or_match_9193) {
        ctx->error = msgprintf("Error at\n%s\n%s\n",
                               " |-> matrix-inversion.fut:43:1-44:33\n |-> matrix-inversion.fut:44:3-33\n `-> matrix-inversion.fut:44:14-29\n",
                               "function arguments of wrong shape");
        if (memblock_unref_device(ctx, &out_mem_10111, "out_mem_10111") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_10006 = sext_i32_i64(sizze_9184);
    int64_t binop_x_10008 = binop_x_10006 * binop_x_10006;
    int64_t bytes_10005 = 4 * binop_x_10008;
    struct memblock_device mem_10009;
    
    mem_10009.references = NULL;
    if (memblock_alloc_device(ctx, &mem_10009, bytes_10005, "mem_10009"))
        return 1;
    
    int call_ret_10180 = futrts__replicate_f32(ctx, mem_10009, sizze_9184 *
                                               sizze_9184, 0.0F);
    
    assert(call_ret_10180 == 0);
    
    int32_t segmap_group_sizze_9336;
    
    segmap_group_sizze_9336 = ctx->sizes.mainzisegmap_group_sizze_9307;
    
    int64_t segmap_group_sizze_9337 = sext_i32_i64(segmap_group_sizze_9336);
    int64_t y_9338 = segmap_group_sizze_9337 - 1;
    int64_t x_9339 = y_9338 + binop_x_10006;
    int64_t segmap_usable_groups_64_9341 = squot64(x_9339,
                                                   segmap_group_sizze_9337);
    int32_t segmap_usable_groups_9342 =
            sext_i64_i32(segmap_usable_groups_64_9341);
    
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9302, 0,
                                            sizeof(sizze_9184), &sizze_9184));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9302, 1,
                                            sizeof(mem_10009.mem),
                                            &mem_10009.mem));
    if (1 * (segmap_usable_groups_9342 * segmap_group_sizze_9336) != 0) {
        const size_t global_work_sizze_10181[1] = {segmap_usable_groups_9342 *
                     segmap_group_sizze_9336};
        const size_t local_work_sizze_10185[1] = {segmap_group_sizze_9336};
        int64_t time_start_10182 = 0, time_end_10183 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "segmap_9302");
            fprintf(stderr, "%zu", global_work_sizze_10181[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_10185[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_10182 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->segmap_9302, 1,
                                                        NULL,
                                                        global_work_sizze_10181,
                                                        local_work_sizze_10185,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->segmap_9302_runs,
                                                                                                        &ctx->segmap_9302_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_10183 = get_wall_time();
            
            long time_diff_10184 = time_end_10183 - time_start_10182;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_9302",
                    time_diff_10184);
        }
    }
    
    int32_t conc_tmp_9220 = sizze_9184 + sizze_9184;
    int32_t y_9236 = sizze_9184 - 1;
    int32_t one_intra_par_min_9374 = sizze_9184 * conc_tmp_9220;
    int32_t intra_avail_par_9376 = smin32(sizze_9184, one_intra_par_min_9374);
    int32_t computed_group_sizze_9350 = smax32(sizze_9184,
                                               one_intra_par_min_9374);
    int32_t max_group_sizze_9439;
    
    max_group_sizze_9439 = ctx->opencl.max_group_size;
    
    bool fits_9440 = sle32(computed_group_sizze_9350, max_group_sizze_9439);
    bool suff_intra_par_9438;
    
    suff_intra_par_9438 = ctx->sizes.mainzisuff_intra_par_2 <=
        intra_avail_par_9376;
    if (ctx->logging)
        fprintf(stderr, "Compared %s <= %d.\n", "main.suff_intra_par_2",
                intra_avail_par_9376);
    
    bool intra_suff_and_fits_9441 = suff_intra_par_9438 && fits_9440;
    int32_t binop_x_10011 = sizze_9184 * sizze_9185;
    int32_t convop_x_10012 = sizze_9183 * binop_x_10011;
    int64_t binop_x_10013 = sext_i32_i64(convop_x_10012);
    int64_t bytes_10010 = 4 * binop_x_10013;
    int64_t binop_x_10036 = sext_i32_i64(sizze_9183);
    int64_t binop_x_10038 = binop_x_10006 * binop_x_10036;
    int64_t binop_x_10040 = binop_x_10006 * binop_x_10038;
    int64_t bytes_10035 = 4 * binop_x_10040;
    int64_t binop_y_10053 = sext_i32_i64(conc_tmp_9220);
    int64_t binop_x_10054 = binop_x_10038 * binop_y_10053;
    int64_t bytes_10049 = 4 * binop_x_10054;
    int64_t bytes_10061 = 4 * binop_x_10038;
    struct memblock_device res_mem_10086;
    
    res_mem_10086.references = NULL;
    if (intra_suff_and_fits_9441) {
        struct memblock_device mem_10014;
        
        mem_10014.references = NULL;
        if (memblock_alloc_device(ctx, &mem_10014, bytes_10010, "mem_10014"))
            return 1;
        
        int call_ret_10186 = futrts__map_transpose_f32(ctx, mem_10014, 0,
                                                       xs_mem_10004, 0, 1,
                                                       sizze_9184 * sizze_9185,
                                                       sizze_9183, sizze_9183 *
                                                       sizze_9184 * sizze_9185,
                                                       sizze_9183 * sizze_9184 *
                                                       sizze_9185);
        
        assert(call_ret_10186 == 0);
        
        struct memblock_device mem_10041;
        
        mem_10041.references = NULL;
        if (memblock_alloc_device(ctx, &mem_10041, bytes_10035, "mem_10041"))
            return 1;
        
        int64_t binop_x_10018 = binop_x_10006 * binop_y_10053;
        int64_t bytes_10015 = 4 * binop_x_10018;
        int64_t bytes_10023 = 4 * binop_x_10006;
        int32_t convop_x_10097 = sizze_9184 * conc_tmp_9220;
        int64_t binop_x_10098 = sext_i32_i64(convop_x_10097);
        int64_t double_buffer_sizze_10099 = 4 * binop_x_10098;
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 0,
                                                bytes_10015, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 1,
                                                binop_x_10006, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 2,
                                                bytes_10023, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 3,
                                                bytes_10023, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 4,
                                                double_buffer_sizze_10099,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 5,
                                                bytes_10015, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 6,
                                                sizeof(sizze_9183),
                                                &sizze_9183));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 7,
                                                sizeof(sizze_9184),
                                                &sizze_9184));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 8,
                                                sizeof(sizze_9185),
                                                &sizze_9185));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 9,
                                                sizeof(conc_tmp_9220),
                                                &conc_tmp_9220));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 10,
                                                sizeof(y_9236), &y_9236));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 11,
                                                sizeof(computed_group_sizze_9350),
                                                &computed_group_sizze_9350));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 12,
                                                sizeof(mem_10009.mem),
                                                &mem_10009.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 13,
                                                sizeof(mem_10014.mem),
                                                &mem_10014.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_intragroup_9377, 14,
                                                sizeof(mem_10041.mem),
                                                &mem_10041.mem));
        if (1 * (sizze_9183 * computed_group_sizze_9350) != 0) {
            const size_t global_work_sizze_10187[1] = {sizze_9183 *
                         computed_group_sizze_9350};
            const size_t local_work_sizze_10191[1] =
                         {computed_group_sizze_9350};
            int64_t time_start_10188 = 0, time_end_10189 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "segmap_intragroup_9377");
                fprintf(stderr, "%zu", global_work_sizze_10187[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_10191[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + bytes_10015 + binop_x_10006 + bytes_10023 +
                               bytes_10023 + double_buffer_sizze_10099 +
                               bytes_10015));
                time_start_10188 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->segmap_intragroup_9377,
                                                            1, NULL,
                                                            global_work_sizze_10187,
                                                            local_work_sizze_10191,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->segmap_intragroup_9377_runs,
                                                                                                            &ctx->segmap_intragroup_9377_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_10189 = get_wall_time();
                
                long time_diff_10190 = time_end_10189 - time_start_10188;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "segmap_intragroup_9377", time_diff_10190);
            }
        }
        if (memblock_unref_device(ctx, &mem_10014, "mem_10014") != 0)
            return 1;
        if (memblock_set_device(ctx, &res_mem_10086, &mem_10041, "mem_10041") !=
            0)
            return 1;
        if (memblock_unref_device(ctx, &mem_10041, "mem_10041") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_10014, "mem_10014") != 0)
            return 1;
    } else {
        struct memblock_device mem_10048;
        
        mem_10048.references = NULL;
        if (memblock_alloc_device(ctx, &mem_10048, bytes_10035, "mem_10048"))
            return 1;
        
        int32_t group_sizze_10146;
        
        group_sizze_10146 = ctx->sizes.mainzigroup_sizze_10146;
        
        int32_t num_groups_10147;
        
        num_groups_10147 = squot32(sizze_9183 * sizze_9184 * sizze_9184 +
                                   sext_i32_i32(group_sizze_10146) - 1,
                                   sext_i32_i32(group_sizze_10146));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_10143, 0,
                                                sizeof(sizze_9183),
                                                &sizze_9183));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_10143, 1,
                                                sizeof(sizze_9184),
                                                &sizze_9184));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_10143, 2,
                                                sizeof(mem_10009.mem),
                                                &mem_10009.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_10143, 3,
                                                sizeof(mem_10048.mem),
                                                &mem_10048.mem));
        if (1 * (num_groups_10147 * group_sizze_10146) != 0) {
            const size_t global_work_sizze_10192[1] = {num_groups_10147 *
                         group_sizze_10146};
            const size_t local_work_sizze_10196[1] = {group_sizze_10146};
            int64_t time_start_10193 = 0, time_end_10194 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "replicate_10143");
                fprintf(stderr, "%zu", global_work_sizze_10192[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_10196[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_10193 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->replicate_10143,
                                                            1, NULL,
                                                            global_work_sizze_10192,
                                                            local_work_sizze_10196,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->replicate_10143_runs,
                                                                                                            &ctx->replicate_10143_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_10194 = get_wall_time();
                
                long time_diff_10195 = time_end_10194 - time_start_10193;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n", "replicate_10143",
                        time_diff_10195);
            }
        }
        
        struct memblock_device mem_10055;
        
        mem_10055.references = NULL;
        if (memblock_alloc_device(ctx, &mem_10055, bytes_10049, "mem_10055"))
            return 1;
        
        int32_t tmp_offs_10148 = 0;
        int32_t group_sizze_10152;
        
        group_sizze_10152 = ctx->sizes.mainzigroup_sizze_10152;
        
        int32_t num_groups_10153;
        
        num_groups_10153 = squot32(sizze_9183 * (sizze_9184 * sizze_9184) +
                                   sext_i32_i32(group_sizze_10152) - 1,
                                   sext_i32_i32(group_sizze_10152));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10149, 0,
                                                sizeof(sizze_9183),
                                                &sizze_9183));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10149, 1,
                                                sizeof(sizze_9184),
                                                &sizze_9184));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10149, 2,
                                                sizeof(conc_tmp_9220),
                                                &conc_tmp_9220));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10149, 3,
                                                sizeof(xs_mem_10004.mem),
                                                &xs_mem_10004.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10149, 4,
                                                sizeof(mem_10055.mem),
                                                &mem_10055.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10149, 5,
                                                sizeof(tmp_offs_10148),
                                                &tmp_offs_10148));
        if (1 * (num_groups_10153 * group_sizze_10152) != 0) {
            const size_t global_work_sizze_10197[1] = {num_groups_10153 *
                         group_sizze_10152};
            const size_t local_work_sizze_10201[1] = {group_sizze_10152};
            int64_t time_start_10198 = 0, time_end_10199 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "copy_10149");
                fprintf(stderr, "%zu", global_work_sizze_10197[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_10201[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_10198 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->copy_10149, 1,
                                                            NULL,
                                                            global_work_sizze_10197,
                                                            local_work_sizze_10201,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->copy_10149_runs,
                                                                                                            &ctx->copy_10149_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_10199 = get_wall_time();
                
                long time_diff_10200 = time_end_10199 - time_start_10198;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n", "copy_10149",
                        time_diff_10200);
            }
        }
        tmp_offs_10148 += sizze_9184;
        
        int32_t group_sizze_10157;
        
        group_sizze_10157 = ctx->sizes.mainzigroup_sizze_10157;
        
        int32_t num_groups_10158;
        
        num_groups_10158 = squot32(sizze_9183 * (sizze_9184 * sizze_9184) +
                                   sext_i32_i32(group_sizze_10157) - 1,
                                   sext_i32_i32(group_sizze_10157));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10154, 0,
                                                sizeof(sizze_9183),
                                                &sizze_9183));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10154, 1,
                                                sizeof(sizze_9184),
                                                &sizze_9184));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10154, 2,
                                                sizeof(conc_tmp_9220),
                                                &conc_tmp_9220));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10154, 3,
                                                sizeof(mem_10048.mem),
                                                &mem_10048.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10154, 4,
                                                sizeof(mem_10055.mem),
                                                &mem_10055.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10154, 5,
                                                sizeof(tmp_offs_10148),
                                                &tmp_offs_10148));
        if (1 * (num_groups_10158 * group_sizze_10157) != 0) {
            const size_t global_work_sizze_10202[1] = {num_groups_10158 *
                         group_sizze_10157};
            const size_t local_work_sizze_10206[1] = {group_sizze_10157};
            int64_t time_start_10203 = 0, time_end_10204 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "copy_10154");
                fprintf(stderr, "%zu", global_work_sizze_10202[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_10206[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_10203 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->copy_10154, 1,
                                                            NULL,
                                                            global_work_sizze_10202,
                                                            local_work_sizze_10206,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->copy_10154_runs,
                                                                                                            &ctx->copy_10154_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_10204 = get_wall_time();
                
                long time_diff_10205 = time_end_10204 - time_start_10203;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n", "copy_10154",
                        time_diff_10205);
            }
        }
        tmp_offs_10148 += sizze_9184;
        if (memblock_unref_device(ctx, &mem_10048, "mem_10048") != 0)
            return 1;
        
        bool loop_nonempty_9983 = slt32(0, sizze_9184);
        int64_t nest_sizze_9890 = binop_x_10006 * binop_x_10036;
        int32_t segmap_group_sizze_9891;
        
        segmap_group_sizze_9891 = ctx->sizes.mainzisegmap_group_sizze_9669;
        
        int64_t segmap_group_sizze_9892 = sext_i32_i64(segmap_group_sizze_9891);
        int64_t y_9893 = segmap_group_sizze_9892 - 1;
        int64_t x_9894 = nest_sizze_9890 + y_9893;
        int64_t segmap_usable_groups_64_9896;
        
        if (loop_nonempty_9983) {
            int64_t x_9984 = squot64(x_9894, segmap_group_sizze_9892);
            
            segmap_usable_groups_64_9896 = x_9984;
        } else {
            segmap_usable_groups_64_9896 = 0;
        }
        
        int32_t segmap_usable_groups_9897 =
                sext_i64_i32(segmap_usable_groups_64_9896);
        int64_t y_9928 = binop_x_10006 * binop_y_10053;
        int64_t nest_sizze_9929 = y_9928 * binop_x_10036;
        int32_t segmap_group_sizze_9930;
        
        segmap_group_sizze_9930 = ctx->sizes.mainzisegmap_group_sizze_9616;
        
        int64_t segmap_group_sizze_9931 = sext_i32_i64(segmap_group_sizze_9930);
        int64_t y_9932 = segmap_group_sizze_9931 - 1;
        int64_t x_9933 = nest_sizze_9929 + y_9932;
        int64_t segmap_usable_groups_64_9935;
        
        if (loop_nonempty_9983) {
            int64_t x_9986 = squot64(x_9933, segmap_group_sizze_9931);
            
            segmap_usable_groups_64_9935 = x_9986;
        } else {
            segmap_usable_groups_64_9935 = 0;
        }
        
        int32_t segmap_usable_groups_9936 =
                sext_i64_i32(segmap_usable_groups_64_9935);
        struct memblock_device mem_10060;
        
        mem_10060.references = NULL;
        if (memblock_alloc_device(ctx, &mem_10060, binop_x_10038, "mem_10060"))
            return 1;
        
        struct memblock_device mem_10065;
        
        mem_10065.references = NULL;
        if (memblock_alloc_device(ctx, &mem_10065, bytes_10061, "mem_10065"))
            return 1;
        
        struct memblock_device mem_10070;
        
        mem_10070.references = NULL;
        if (memblock_alloc_device(ctx, &mem_10070, bytes_10061, "mem_10070"))
            return 1;
        
        struct memblock_device res_r_mem_10078;
        
        res_r_mem_10078.references = NULL;
        
        struct memblock_device A_expanded_mem_10056;
        
        A_expanded_mem_10056.references = NULL;
        if (memblock_set_device(ctx, &A_expanded_mem_10056, &mem_10055,
                                "mem_10055") != 0)
            return 1;
        for (int32_t i_9862 = 0; i_9862 < sizze_9184; i_9862++) {
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9664, 0,
                                                    sizeof(sizze_9183),
                                                    &sizze_9183));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9664, 1,
                                                    sizeof(sizze_9184),
                                                    &sizze_9184));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9664, 2,
                                                    sizeof(conc_tmp_9220),
                                                    &conc_tmp_9220));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9664, 3,
                                                    sizeof(y_9236), &y_9236));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9664, 4,
                                                    sizeof(i_9862), &i_9862));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9664, 5,
                                                    sizeof(A_expanded_mem_10056.mem),
                                                    &A_expanded_mem_10056.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9664, 6,
                                                    sizeof(mem_10060.mem),
                                                    &mem_10060.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9664, 7,
                                                    sizeof(mem_10065.mem),
                                                    &mem_10065.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9664, 8,
                                                    sizeof(mem_10070.mem),
                                                    &mem_10070.mem));
            if (1 * (segmap_usable_groups_9897 * segmap_group_sizze_9891) !=
                0) {
                const size_t global_work_sizze_10207[1] =
                             {segmap_usable_groups_9897 *
                             segmap_group_sizze_9891};
                const size_t local_work_sizze_10211[1] =
                             {segmap_group_sizze_9891};
                int64_t time_start_10208 = 0, time_end_10209 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "segmap_9664");
                    fprintf(stderr, "%zu", global_work_sizze_10207[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_10211[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) 0);
                    time_start_10208 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->segmap_9664,
                                                                1, NULL,
                                                                global_work_sizze_10207,
                                                                local_work_sizze_10211,
                                                                0, NULL,
                                                                ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                &ctx->segmap_9664_runs,
                                                                                                                &ctx->segmap_9664_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_10209 = get_wall_time();
                    
                    long time_diff_10210 = time_end_10209 - time_start_10208;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_9664",
                            time_diff_10210);
                }
            }
            
            struct memblock_device mem_10077;
            
            mem_10077.references = NULL;
            if (memblock_alloc_device(ctx, &mem_10077, bytes_10049,
                                      "mem_10077"))
                return 1;
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9609, 0,
                                                    sizeof(sizze_9183),
                                                    &sizze_9183));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9609, 1,
                                                    sizeof(sizze_9184),
                                                    &sizze_9184));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9609, 2,
                                                    sizeof(conc_tmp_9220),
                                                    &conc_tmp_9220));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9609, 3,
                                                    sizeof(y_9236), &y_9236));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9609, 4,
                                                    sizeof(i_9862), &i_9862));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9609, 5,
                                                    sizeof(A_expanded_mem_10056.mem),
                                                    &A_expanded_mem_10056.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9609, 6,
                                                    sizeof(mem_10070.mem),
                                                    &mem_10070.mem));
            OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_9609, 7,
                                                    sizeof(mem_10077.mem),
                                                    &mem_10077.mem));
            if (1 * (segmap_usable_groups_9936 * segmap_group_sizze_9930) !=
                0) {
                const size_t global_work_sizze_10212[1] =
                             {segmap_usable_groups_9936 *
                             segmap_group_sizze_9930};
                const size_t local_work_sizze_10216[1] =
                             {segmap_group_sizze_9930};
                int64_t time_start_10213 = 0, time_end_10214 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with global work size [",
                            "segmap_9609");
                    fprintf(stderr, "%zu", global_work_sizze_10212[0]);
                    fprintf(stderr, "] and local work size [");
                    fprintf(stderr, "%zu", local_work_sizze_10216[0]);
                    fprintf(stderr,
                            "]; local memory parameters sum to %d bytes.\n",
                            (int) 0);
                    time_start_10213 = get_wall_time();
                }
                OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                ctx->segmap_9609,
                                                                1, NULL,
                                                                global_work_sizze_10212,
                                                                local_work_sizze_10216,
                                                                0, NULL,
                                                                ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                &ctx->segmap_9609_runs,
                                                                                                                &ctx->segmap_9609_total_runtime)));
                if (ctx->debugging) {
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                    time_end_10214 = get_wall_time();
                    
                    long time_diff_10215 = time_end_10214 - time_start_10213;
                    
                    fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_9609",
                            time_diff_10215);
                }
            }
            
            struct memblock_device A_expanded_mem_tmp_10159;
            
            A_expanded_mem_tmp_10159.references = NULL;
            if (memblock_set_device(ctx, &A_expanded_mem_tmp_10159, &mem_10077,
                                    "mem_10077") != 0)
                return 1;
            if (memblock_set_device(ctx, &A_expanded_mem_10056,
                                    &A_expanded_mem_tmp_10159,
                                    "A_expanded_mem_tmp_10159") != 0)
                return 1;
            if (memblock_unref_device(ctx, &A_expanded_mem_tmp_10159,
                                      "A_expanded_mem_tmp_10159") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_10077, "mem_10077") != 0)
                return 1;
        }
        if (memblock_set_device(ctx, &res_r_mem_10078, &A_expanded_mem_10056,
                                "A_expanded_mem_10056") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_10055, "mem_10055") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_10060, "mem_10060") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_10065, "mem_10065") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_10070, "mem_10070") != 0)
            return 1;
        
        struct memblock_device mem_10085;
        
        mem_10085.references = NULL;
        if (memblock_alloc_device(ctx, &mem_10085, bytes_10035, "mem_10085"))
            return 1;
        
        int32_t group_sizze_10174;
        
        group_sizze_10174 = ctx->sizes.mainzigroup_sizze_10174;
        
        int32_t num_groups_10175;
        
        num_groups_10175 = squot32(sizze_9183 * (sizze_9184 * sizze_9184) +
                                   sext_i32_i32(group_sizze_10174) - 1,
                                   sext_i32_i32(group_sizze_10174));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10171, 0,
                                                sizeof(sizze_9183),
                                                &sizze_9183));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10171, 1,
                                                sizeof(sizze_9184),
                                                &sizze_9184));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10171, 2,
                                                sizeof(conc_tmp_9220),
                                                &conc_tmp_9220));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10171, 3,
                                                sizeof(res_r_mem_10078.mem),
                                                &res_r_mem_10078.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->copy_10171, 4,
                                                sizeof(mem_10085.mem),
                                                &mem_10085.mem));
        if (1 * (num_groups_10175 * group_sizze_10174) != 0) {
            const size_t global_work_sizze_10217[1] = {num_groups_10175 *
                         group_sizze_10174};
            const size_t local_work_sizze_10221[1] = {group_sizze_10174};
            int64_t time_start_10218 = 0, time_end_10219 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "copy_10171");
                fprintf(stderr, "%zu", global_work_sizze_10217[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_10221[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_10218 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->copy_10171, 1,
                                                            NULL,
                                                            global_work_sizze_10217,
                                                            local_work_sizze_10221,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->copy_10171_runs,
                                                                                                            &ctx->copy_10171_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_10219 = get_wall_time();
                
                long time_diff_10220 = time_end_10219 - time_start_10218;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n", "copy_10171",
                        time_diff_10220);
            }
        }
        if (memblock_unref_device(ctx, &res_r_mem_10078, "res_r_mem_10078") !=
            0)
            return 1;
        if (memblock_set_device(ctx, &res_mem_10086, &mem_10085, "mem_10085") !=
            0)
            return 1;
        if (memblock_unref_device(ctx, &mem_10085, "mem_10085") != 0)
            return 1;
        if (memblock_unref_device(ctx, &A_expanded_mem_10056,
                                  "A_expanded_mem_10056") != 0)
            return 1;
        if (memblock_unref_device(ctx, &res_r_mem_10078, "res_r_mem_10078") !=
            0)
            return 1;
        if (memblock_unref_device(ctx, &mem_10070, "mem_10070") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_10065, "mem_10065") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_10060, "mem_10060") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_10055, "mem_10055") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_10048, "mem_10048") != 0)
            return 1;
    }
    if (memblock_unref_device(ctx, &mem_10009, "mem_10009") != 0)
        return 1;
    out_arrsizze_10112 = sizze_9183;
    out_arrsizze_10113 = sizze_9184;
    out_arrsizze_10114 = sizze_9184;
    if (memblock_set_device(ctx, &out_mem_10111, &res_mem_10086,
                            "res_mem_10086") != 0)
        return 1;
    (*out_mem_p_10176).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_10176, &out_mem_10111,
                            "out_mem_10111") != 0)
        return 1;
    *out_out_arrsizze_10177 = out_arrsizze_10112;
    *out_out_arrsizze_10178 = out_arrsizze_10113;
    *out_out_arrsizze_10179 = out_arrsizze_10114;
    if (memblock_unref_device(ctx, &res_mem_10086, "res_mem_10086") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_10009, "mem_10009") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_10111, "out_mem_10111") != 0)
        return 1;
    return 0;
}
static int futrts__map_transpose_f32(struct futhark_context *ctx,
                                     struct memblock_device destmem_0,
                                     int32_t destoffset_1,
                                     struct memblock_device srcmem_2,
                                     int32_t srcoffset_3, int32_t num_arrays_4,
                                     int32_t x_elems_5, int32_t y_elems_6,
                                     int32_t in_elems_7, int32_t out_elems_8)
{
    if (!(num_arrays_4 == 0 || (x_elems_5 == 0 || y_elems_6 == 0))) {
        int32_t muly_10 = squot32(16, x_elems_5);
        int32_t mulx_9 = squot32(16, y_elems_6);
        
        if (in_elems_7 == out_elems_8 && ((num_arrays_4 == 1 || x_elems_5 *
                                           y_elems_6 == in_elems_7) &&
                                          (x_elems_5 == 1 || y_elems_6 == 1))) {
            if (in_elems_7 * sizeof(float) > 0) {
                OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                             srcmem_2.mem,
                                                             destmem_0.mem,
                                                             srcoffset_3,
                                                             destoffset_1,
                                                             in_elems_7 *
                                                             sizeof(float), 0,
                                                             NULL,
                                                             ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                             &ctx->copy_dev_to_dev_runs,
                                                                                                             &ctx->copy_dev_to_dev_total_runtime)));
                if (ctx->debugging)
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            }
        } else {
            if (sle32(x_elems_5, 8) && slt32(16, y_elems_6)) {
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        0, 1088, NULL));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        1, sizeof(destoffset_1),
                                                        &destoffset_1));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        2, sizeof(srcoffset_3),
                                                        &srcoffset_3));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        3, sizeof(num_arrays_4),
                                                        &num_arrays_4));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        4, sizeof(x_elems_5),
                                                        &x_elems_5));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        5, sizeof(y_elems_6),
                                                        &y_elems_6));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        6, sizeof(in_elems_7),
                                                        &in_elems_7));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        7, sizeof(out_elems_8),
                                                        &out_elems_8));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        8, sizeof(mulx_9),
                                                        &mulx_9));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        9, sizeof(muly_10),
                                                        &muly_10));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        10,
                                                        sizeof(destmem_0.mem),
                                                        &destmem_0.mem));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_width,
                                                        11,
                                                        sizeof(srcmem_2.mem),
                                                        &srcmem_2.mem));
                if (1 * (squot32(x_elems_5 + 16 - 1, 16) * 16) *
                    (squot32(squot32(y_elems_6 + muly_10 - 1, muly_10) + 16 - 1,
                             16) * 16) * (num_arrays_4 * 1) != 0) {
                    const size_t global_work_sizze_10222[3] =
                                 {squot32(x_elems_5 + 16 - 1, 16) * 16,
                                  squot32(squot32(y_elems_6 + muly_10 - 1,
                                                  muly_10) + 16 - 1, 16) * 16,
                                  num_arrays_4 * 1};
                    const size_t local_work_sizze_10226[3] = {16, 16, 1};
                    int64_t time_start_10223 = 0, time_end_10224 = 0;
                    
                    if (ctx->debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "map_transpose_f32_low_width");
                        fprintf(stderr, "%zu", global_work_sizze_10222[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_10222[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_10222[2]);
                        fprintf(stderr, "] and local work size [");
                        fprintf(stderr, "%zu", local_work_sizze_10226[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_10226[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_10226[2]);
                        fprintf(stderr,
                                "]; local memory parameters sum to %d bytes.\n",
                                (int) (0 + 1088));
                        time_start_10223 = get_wall_time();
                    }
                    OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                    ctx->map_transpose_f32_low_width,
                                                                    3, NULL,
                                                                    global_work_sizze_10222,
                                                                    local_work_sizze_10226,
                                                                    0, NULL,
                                                                    ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                    &ctx->map_transpose_f32_low_width_runs,
                                                                                                                    &ctx->map_transpose_f32_low_width_total_runtime)));
                    if (ctx->debugging) {
                        OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                        time_end_10224 = get_wall_time();
                        
                        long time_diff_10225 = time_end_10224 -
                             time_start_10223;
                        
                        fprintf(stderr, "kernel %s runtime: %ldus\n",
                                "map_transpose_f32_low_width", time_diff_10225);
                    }
                }
            } else {
                if (sle32(y_elems_6, 8) && slt32(16, x_elems_5)) {
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            0, 1088, NULL));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            1,
                                                            sizeof(destoffset_1),
                                                            &destoffset_1));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            2,
                                                            sizeof(srcoffset_3),
                                                            &srcoffset_3));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            3,
                                                            sizeof(num_arrays_4),
                                                            &num_arrays_4));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            4,
                                                            sizeof(x_elems_5),
                                                            &x_elems_5));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            5,
                                                            sizeof(y_elems_6),
                                                            &y_elems_6));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            6,
                                                            sizeof(in_elems_7),
                                                            &in_elems_7));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            7,
                                                            sizeof(out_elems_8),
                                                            &out_elems_8));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            8, sizeof(mulx_9),
                                                            &mulx_9));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            9, sizeof(muly_10),
                                                            &muly_10));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            10,
                                                            sizeof(destmem_0.mem),
                                                            &destmem_0.mem));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_low_height,
                                                            11,
                                                            sizeof(srcmem_2.mem),
                                                            &srcmem_2.mem));
                    if (1 * (squot32(squot32(x_elems_5 + mulx_9 - 1, mulx_9) +
                                     16 - 1, 16) * 16) * (squot32(y_elems_6 +
                                                                  16 - 1, 16) *
                                                          16) * (num_arrays_4 *
                                                                 1) != 0) {
                        const size_t global_work_sizze_10227[3] =
                                     {squot32(squot32(x_elems_5 + mulx_9 - 1,
                                                      mulx_9) + 16 - 1, 16) *
                                      16, squot32(y_elems_6 + 16 - 1, 16) * 16,
                                      num_arrays_4 * 1};
                        const size_t local_work_sizze_10231[3] = {16, 16, 1};
                        int64_t time_start_10228 = 0, time_end_10229 = 0;
                        
                        if (ctx->debugging) {
                            fprintf(stderr,
                                    "Launching %s with global work size [",
                                    "map_transpose_f32_low_height");
                            fprintf(stderr, "%zu", global_work_sizze_10227[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_10227[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_10227[2]);
                            fprintf(stderr, "] and local work size [");
                            fprintf(stderr, "%zu", local_work_sizze_10231[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_10231[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_10231[2]);
                            fprintf(stderr,
                                    "]; local memory parameters sum to %d bytes.\n",
                                    (int) (0 + 1088));
                            time_start_10228 = get_wall_time();
                        }
                        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                        ctx->map_transpose_f32_low_height,
                                                                        3, NULL,
                                                                        global_work_sizze_10227,
                                                                        local_work_sizze_10231,
                                                                        0, NULL,
                                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                        &ctx->map_transpose_f32_low_height_runs,
                                                                                                                        &ctx->map_transpose_f32_low_height_total_runtime)));
                        if (ctx->debugging) {
                            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                            time_end_10229 = get_wall_time();
                            
                            long time_diff_10230 = time_end_10229 -
                                 time_start_10228;
                            
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "map_transpose_f32_low_height",
                                    time_diff_10230);
                        }
                    }
                } else {
                    if (sle32(x_elems_5, 8) && sle32(y_elems_6, 8)) {
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                0, 1, NULL));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                1,
                                                                sizeof(destoffset_1),
                                                                &destoffset_1));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                2,
                                                                sizeof(srcoffset_3),
                                                                &srcoffset_3));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                3,
                                                                sizeof(num_arrays_4),
                                                                &num_arrays_4));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                4,
                                                                sizeof(x_elems_5),
                                                                &x_elems_5));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                5,
                                                                sizeof(y_elems_6),
                                                                &y_elems_6));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                6,
                                                                sizeof(in_elems_7),
                                                                &in_elems_7));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                7,
                                                                sizeof(out_elems_8),
                                                                &out_elems_8));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                8,
                                                                sizeof(mulx_9),
                                                                &mulx_9));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                9,
                                                                sizeof(muly_10),
                                                                &muly_10));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                10,
                                                                sizeof(destmem_0.mem),
                                                                &destmem_0.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32_small,
                                                                11,
                                                                sizeof(srcmem_2.mem),
                                                                &srcmem_2.mem));
                        if (1 * (squot32(num_arrays_4 * x_elems_5 * y_elems_6 +
                                         256 - 1, 256) * 256) != 0) {
                            const size_t global_work_sizze_10232[1] =
                                         {squot32(num_arrays_4 * x_elems_5 *
                                                  y_elems_6 + 256 - 1, 256) *
                                         256};
                            const size_t local_work_sizze_10236[1] = {256};
                            int64_t time_start_10233 = 0, time_end_10234 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "map_transpose_f32_small");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_10232[0]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_10236[0]);
                                fprintf(stderr,
                                        "]; local memory parameters sum to %d bytes.\n",
                                        (int) (0 + 1));
                                time_start_10233 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->map_transpose_f32_small,
                                                                            1,
                                                                            NULL,
                                                                            global_work_sizze_10232,
                                                                            local_work_sizze_10236,
                                                                            0,
                                                                            NULL,
                                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                            &ctx->map_transpose_f32_small_runs,
                                                                                                                            &ctx->map_transpose_f32_small_total_runtime)));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_10234 = get_wall_time();
                                
                                long time_diff_10235 = time_end_10234 -
                                     time_start_10233;
                                
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "map_transpose_f32_small",
                                        time_diff_10235);
                            }
                        }
                    } else {
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                0, 4224, NULL));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                1,
                                                                sizeof(destoffset_1),
                                                                &destoffset_1));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                2,
                                                                sizeof(srcoffset_3),
                                                                &srcoffset_3));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                3,
                                                                sizeof(num_arrays_4),
                                                                &num_arrays_4));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                4,
                                                                sizeof(x_elems_5),
                                                                &x_elems_5));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                5,
                                                                sizeof(y_elems_6),
                                                                &y_elems_6));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                6,
                                                                sizeof(in_elems_7),
                                                                &in_elems_7));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                7,
                                                                sizeof(out_elems_8),
                                                                &out_elems_8));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                8,
                                                                sizeof(mulx_9),
                                                                &mulx_9));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                9,
                                                                sizeof(muly_10),
                                                                &muly_10));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                10,
                                                                sizeof(destmem_0.mem),
                                                                &destmem_0.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_f32,
                                                                11,
                                                                sizeof(srcmem_2.mem),
                                                                &srcmem_2.mem));
                        if (1 * (squot32(x_elems_5 + 32 - 1, 32) * 32) *
                            (squot32(y_elems_6 + 32 - 1, 32) * 8) *
                            (num_arrays_4 * 1) != 0) {
                            const size_t global_work_sizze_10237[3] =
                                         {squot32(x_elems_5 + 32 - 1, 32) * 32,
                                          squot32(y_elems_6 + 32 - 1, 32) * 8,
                                          num_arrays_4 * 1};
                            const size_t local_work_sizze_10241[3] = {32, 8, 1};
                            int64_t time_start_10238 = 0, time_end_10239 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "map_transpose_f32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_10237[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_10237[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_10237[2]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_10241[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_10241[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_10241[2]);
                                fprintf(stderr,
                                        "]; local memory parameters sum to %d bytes.\n",
                                        (int) (0 + 4224));
                                time_start_10238 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->map_transpose_f32,
                                                                            3,
                                                                            NULL,
                                                                            global_work_sizze_10237,
                                                                            local_work_sizze_10241,
                                                                            0,
                                                                            NULL,
                                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                            &ctx->map_transpose_f32_runs,
                                                                                                                            &ctx->map_transpose_f32_total_runtime)));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_10239 = get_wall_time();
                                
                                long time_diff_10240 = time_end_10239 -
                                     time_start_10238;
                                
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "map_transpose_f32", time_diff_10240);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
static int futrts__replicate_f32(struct futhark_context *ctx,
                                 struct memblock_device mem_10115,
                                 int32_t num_elems_10116, float val_10117)
{
    int32_t group_sizze_10122;
    
    group_sizze_10122 = ctx->sizes.mainzigroup_sizze_10122;
    
    int32_t num_groups_10123;
    
    num_groups_10123 = squot32(num_elems_10116 +
                               sext_i32_i32(group_sizze_10122) - 1,
                               sext_i32_i32(group_sizze_10122));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_10119, 0,
                                            sizeof(mem_10115.mem),
                                            &mem_10115.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_10119, 1,
                                            sizeof(num_elems_10116),
                                            &num_elems_10116));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_10119, 2,
                                            sizeof(val_10117), &val_10117));
    if (1 * (num_groups_10123 * group_sizze_10122) != 0) {
        const size_t global_work_sizze_10242[1] = {num_groups_10123 *
                     group_sizze_10122};
        const size_t local_work_sizze_10246[1] = {group_sizze_10122};
        int64_t time_start_10243 = 0, time_end_10244 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "replicate_10119");
            fprintf(stderr, "%zu", global_work_sizze_10242[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_10246[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_10243 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->replicate_10119, 1,
                                                        NULL,
                                                        global_work_sizze_10242,
                                                        local_work_sizze_10246,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->replicate_10119_runs,
                                                                                                        &ctx->replicate_10119_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_10244 = get_wall_time();
            
            long time_diff_10245 = time_end_10244 - time_start_10243;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "replicate_10119",
                    time_diff_10245);
        }
    }
    return 0;
}
struct futhark_f32_3d {
    struct memblock_device mem;
    int64_t shape[3];
} ;
struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx,
                                          float *data, int64_t dim0,
                                          int64_t dim1, int64_t dim2)
{
    struct futhark_f32_3d *bad = NULL;
    struct futhark_f32_3d *arr =
                          (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * dim1 * dim2 *
                              sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    if (dim0 * dim1 * dim2 * sizeof(float) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      arr->mem.mem, CL_TRUE, 0,
                                                      dim0 * dim1 * dim2 *
                                                      sizeof(float), data + 0,
                                                      0, NULL,
                                                      ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                      &ctx->copy_dev_to_host_runs,
                                                                                                      &ctx->copy_dev_to_host_total_runtime)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_3d *futhark_new_raw_f32_3d(struct futhark_context *ctx,
                                              cl_mem data, int offset,
                                              int64_t dim0, int64_t dim1,
                                              int64_t dim2)
{
    struct futhark_f32_3d *bad = NULL;
    struct futhark_f32_3d *arr =
                          (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * dim1 * dim2 *
                              sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    if (dim0 * dim1 * dim2 * sizeof(float) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue, data,
                                                     arr->mem.mem, offset, 0,
                                                     dim0 * dim1 * dim2 *
                                                     sizeof(float), 0, NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_dev_to_dev_runs,
                                                                                                     &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_3d(struct futhark_context *ctx,
                          struct futhark_f32_3d *arr, float *data)
{
    lock_lock(&ctx->lock);
    if (arr->shape[0] * arr->shape[1] * arr->shape[2] * sizeof(float) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     arr->mem.mem, CL_TRUE, 0,
                                                     arr->shape[0] *
                                                     arr->shape[1] *
                                                     arr->shape[2] *
                                                     sizeof(float), data + 0, 0,
                                                     NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_host_to_dev_runs,
                                                                                                     &ctx->copy_host_to_dev_total_runtime)));
    lock_unlock(&ctx->lock);
    return 0;
}
cl_mem futhark_values_raw_f32_3d(struct futhark_context *ctx,
                                 struct futhark_f32_3d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_3d(struct futhark_context *ctx,
                              struct futhark_f32_3d *arr)
{
    (void) ctx;
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_3d **out0, const
                       struct futhark_f32_3d *in0)
{
    struct memblock_device xs_mem_10004;
    
    xs_mem_10004.references = NULL;
    
    int32_t sizze_9183;
    int32_t sizze_9184;
    int32_t sizze_9185;
    struct memblock_device out_mem_10111;
    
    out_mem_10111.references = NULL;
    
    int32_t out_arrsizze_10112;
    int32_t out_arrsizze_10113;
    int32_t out_arrsizze_10114;
    
    lock_lock(&ctx->lock);
    xs_mem_10004 = in0->mem;
    sizze_9183 = in0->shape[0];
    sizze_9184 = in0->shape[1];
    sizze_9185 = in0->shape[2];
    
    int ret = futrts_main(ctx, &out_mem_10111, &out_arrsizze_10112,
                          &out_arrsizze_10113, &out_arrsizze_10114,
                          xs_mem_10004, sizze_9183, sizze_9184, sizze_9185);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_3d *) malloc(sizeof(struct futhark_f32_3d))) !=
            NULL);
        (*out0)->mem = out_mem_10111;
        (*out0)->shape[0] = out_arrsizze_10112;
        (*out0)->shape[1] = out_arrsizze_10113;
        (*out0)->shape[2] = out_arrsizze_10114;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
