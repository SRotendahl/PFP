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

struct futhark_f32_1d ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int64_t dim0);
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              cl_mem data, int offset,
                                              int64_t dim0);
int futhark_free_f32_1d(struct futhark_context *ctx,
                        struct futhark_f32_1d *arr);
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data);
cl_mem futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                 struct futhark_f32_1d *arr);
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr);

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_1d **out0, const
                       struct futhark_f32_1d *in0);

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
    
    struct futhark_f32_1d *read_value_19415;
    int64_t read_shape_19416[1];
    float *read_arr_19417 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_19417, read_shape_19416, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_19418;
    
    if (perform_warmup) {
        int r;
        
        assert((read_value_19415 = futhark_new_f32_1d(ctx, read_arr_19417,
                                                      read_shape_19416[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_19418, read_value_19415);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_19415) == 0);
        assert(futhark_free_f32_1d(ctx, result_19418) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        assert((read_value_19415 = futhark_new_f32_1d(ctx, read_arr_19417,
                                                      read_shape_19416[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_19418, read_value_19415);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_19415) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_19418) == 0);
        }
    }
    free(read_arr_19417);
    if (binary_output)
        set_binary_mode(stdout);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_19418)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_19418, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_19418), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_19418) == 0);
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
                   "(x, y);\n}\nstatic inline float futrts_gamma32(float x)\n{\n    return tgamma(x);\n}\nstatic inline float futrts_lgamma32(float x)\n{\n    return lgamma(x);\n}\nstatic inline char futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline char futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#ifdef __OPENCL_VERSION__\nstatic inline float fmod32(float x, float y)\n{\n    return fmod(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floor(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceil(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t)\n{\n    return mix(v0, v1, t);\n}\n#else\nstatic inline float fmod32(float x, float y)\n{\n    return fmodf(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rintf(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floorf(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceilf(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t)\n{\n    return v0 + (v1 - v0) * t;\n}\n#endif\n__kernel void map_transpose_i32(__local volatile\n                                int64_t *block_11_backing_aligned_0,\n                                int32_t destoffset_1, int32_t srcoffset_3,\n                                int32_t num_arrays_4, int32_t x_elems_5,\n                                int32_t y_elems_6, int32_t in_elems_7,\n                                int32_t out_elems_8, int32_t mulx_9,\n                                int32_t muly_10, __global\n                                unsigned char *destmem_0, __global\n                                unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int bloc",
                   "k_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_11_backing_0 = (__local volatile\n                                                          char *) block_11_backing_aligned_0;\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;\n    int32_t x_index_31 = get_global_id_0_37;\n    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;\n    \n    if (slt32(x_index_31, x_elems_5)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +\n                    x_index_31;\n            \n            if (slt32(y_index_32 + j_43 * 8, y_elems_6) && slt32(index_in_35,\n                                                                 in_elems_7)) {\n                ((__local int32_t *) block_11)[(get_local_id_1_39 + j_43 * 8) *\n                                               33 + get_local_id_0_38] =\n                    ((__global int32_t *) srcmem_2)[idata_offset_34 +\n                                                    index_in_35];\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;\n    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;\n    if (slt3",
                   "2(x_index_31, y_elems_6)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +\n                    x_index_31;\n            \n            if (slt32(y_index_32 + j_43 * 8, x_elems_5) && slt32(index_out_36,\n                                                                 out_elems_8)) {\n                ((__global int32_t *) destmem_0)[odata_offset_33 +\n                                                 index_out_36] = ((__local\n                                                                   int32_t *) block_11)[get_local_id_0_38 *\n                                                                                        33 +\n                                                                                        get_local_id_1_39 +\n                                                                                        j_43 *\n                                                                                        8];\n            }\n        }\n    }\n}\n__kernel void map_transpose_i32_low_height(__local volatile\n                                           int64_t *block_11_backing_aligned_0,\n                                           int32_t destoffset_1,\n                                           int32_t srcoffset_3,\n                                           int32_t num_arrays_4,\n                                           int32_t x_elems_5, int32_t y_elems_6,\n                                           int32_t in_elems_7,\n                                           int32_t out_elems_8, int32_t mulx_9,\n                                           int32_t muly_10, __global\n                                           unsigned char *destmem_0, __global\n                                           unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_11_backing_0 = (__local volatile\n                                 ",
                   "                         char *) block_11_backing_aligned_0;\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;\n    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_0_38 +\n            srem32(get_local_id_1_39, mulx_9) * 16;\n    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,\n                                                          mulx_9);\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    \n    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&\n                                         slt32(index_in_35, in_elems_7))) {\n        ((__local int32_t *) block_11)[get_local_id_1_39 * 17 +\n                                       get_local_id_0_38] = ((__global\n                                                              int32_t *) srcmem_2)[idata_offset_34 +\n                                                                                   index_in_35];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_9);\n    y_index_32 = get_group_id_0_40 * 16 * mulx_9 + get_local_id_1_39 +\n        srem32(get_local_id_0_38, mulx_9) * 16;\n    \n    int32_t index_out_36 = y_index_32 * y_elems_6 + x_i",
                   "ndex_31;\n    \n    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&\n                                         slt32(index_out_36, out_elems_8))) {\n        ((__global int32_t *) destmem_0)[odata_offset_33 + index_out_36] =\n            ((__local int32_t *) block_11)[get_local_id_0_38 * 17 +\n                                           get_local_id_1_39];\n    }\n}\n__kernel void map_transpose_i32_low_width(__local volatile\n                                          int64_t *block_11_backing_aligned_0,\n                                          int32_t destoffset_1,\n                                          int32_t srcoffset_3,\n                                          int32_t num_arrays_4,\n                                          int32_t x_elems_5, int32_t y_elems_6,\n                                          int32_t in_elems_7,\n                                          int32_t out_elems_8, int32_t mulx_9,\n                                          int32_t muly_10, __global\n                                          unsigned char *destmem_0, __global\n                                          unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_11_backing_0 = (__local volatile\n                                                          char *) block_11_backing_aligned_0;\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t o",
                   "ur_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;\n    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;\n    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,\n                                                          muly_10);\n    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_10 + get_local_id_1_39 +\n            srem32(get_local_id_0_38, muly_10) * 16;\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    \n    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&\n                                         slt32(index_in_35, in_elems_7))) {\n        ((__local int32_t *) block_11)[get_local_id_1_39 * 17 +\n                                       get_local_id_0_38] = ((__global\n                                                              int32_t *) srcmem_2)[idata_offset_34 +\n                                                                                   index_in_35];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = get_group_id_1_41 * 16 * muly_10 + get_local_id_0_38 +\n        srem32(get_local_id_1_39, muly_10) * 16;\n    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_10);\n    \n    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;\n    \n    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&\n                                         slt32(index_out_36, out_elems_8))) {\n        ((__global int32_t *) destmem_0)[odata_offset_33 + index_out_36] =\n            ((__local int32_t *) block_11)[get_local_id_0_38 * 17 +\n                                           get_local_id_1_39];\n    }\n}\n__kernel void map_transpose_i32_small(__local volatile\n                                      int64_t *block_11_backing_aligned_0,\n                                      int32_t destoffset_1, int32_t srcoffset_3,\n                                      int32_t num_arrays_4, int",
                   "32_t x_elems_5,\n                                      int32_t y_elems_6, int32_t in_elems_7,\n                                      int32_t out_elems_8, int32_t mulx_9,\n                                      int32_t muly_10, __global\n                                      unsigned char *destmem_0, __global\n                                      unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict block_11_backing_0 = (__local volatile\n                                                          char *) block_11_backing_aligned_0;\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *\n                                          x_elems_5) * (y_elems_6 * x_elems_5);\n    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *\n                                        x_elems_5), y_elems_6);\n    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);\n    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;\n    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;\n    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;\n    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;\n    \n    if (slt32(get_global_id_0_37, in_elems_7)) {\n        ((__global int32_t *) destmem_0)[odata_offset_33 + index_out_36] =\n            ((__globa",
                   "l int32_t *) srcmem_2)[idata_offset_34 + index_in_35];\n    }\n}\n__kernel void replicate_18470(__global unsigned char *mem_18466,\n                              int32_t num_elems_18467, int32_t val_18468)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_18470;\n    int32_t replicate_ltid_18471;\n    int32_t replicate_gid_18472;\n    \n    replicate_gtid_18470 = get_global_id(0);\n    replicate_ltid_18471 = get_local_id(0);\n    replicate_gid_18472 = get_group_id(0);\n    if (slt32(replicate_gtid_18470, num_elems_18467)) {\n        ((__global int32_t *) mem_18466)[replicate_gtid_18470] = val_18468;\n    }\n}\n__kernel void replicate_18938(__global unsigned char *mem_18934,\n                              int32_t num_elems_18935, float val_18936)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_18938;\n    int32_t replicate_ltid_18939;\n    int32_t replicate_gid_18940;\n    \n    replicate_gtid_18938 = get_global_id(0);\n    replicate_ltid_18939 = get_local_id(0);\n    replicate_gid_18940 = get_group_id(0);\n    if (slt32(replicate_gtid_18938, num_elems_18935)) {\n        ((__global float *) mem_18934)[replicate_gtid_18938] = val_18936;\n    }\n}\n__kernel void scan_stage1_17841(__local volatile\n                                int64_t *scan_arr_mem_18524_backing_aligned_0,\n                                __local volatile\n                                int64_t *scan_arr_mem_18526_backing_aligned_1,\n                                int32_t sizze_17490, __global\n                                unsigned char *shp_mem_18302, __global\n                                unsigned char *mem_18307, __global\n                                unsigned char *mem_18310,\n                                int32_t num_threads_18512)\n{\n    const int32_t segscan_group_sizze_17836 = mainzisegscan_group_sizze_17835;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const",
                   " int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_18524_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_18524_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_18526_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_18526_backing_aligned_1;\n    int32_t global_tid_18519;\n    int32_t local_tid_18520;\n    int32_t group_sizze_18523;\n    int32_t wave_sizze_18522;\n    int32_t group_tid_18521;\n    \n    global_tid_18519 = get_global_id(0);\n    local_tid_18520 = get_local_id(0);\n    group_sizze_18523 = get_local_size(0);\n    wave_sizze_18522 = LOCKSTEP_WIDTH;\n    group_tid_18521 = get_group_id(0);\n    \n    int32_t phys_tid_17841 = global_tid_18519;\n    __local char *scan_arr_mem_18524;\n    \n    scan_arr_mem_18524 = (__local char *) scan_arr_mem_18524_backing_0;\n    \n    __local char *scan_arr_mem_18526;\n    \n    scan_arr_mem_18526 = (__local char *) scan_arr_mem_18526_backing_1;\n    \n    int32_t x_17499;\n    int32_t x_17500;\n    int32_t x_17501;\n    int32_t x_17502;\n    \n    x_17499 = 0;\n    x_17500 = 0;\n    for (int32_t j_18528 = 0; j_18528 < squot32(sizze_17490 +\n                                                num_threads_18512 - 1,\n                                                num_threads_18512); j_18528++) {\n        int32_t chunk_offset_18529 = segscan_group_sizze_17836 * j_18528 +\n                group_tid_18521 * (segscan_group_sizze_17836 *\n                                   squot32(sizze_17490 + num_threads_18512 - 1,\n                                           num_threads_18512));\n        int32_t flat_idx_18530 = chunk_offset_18529 + local_tid_18520;\n        int32_t gtid_17840 = flat_idx_18530;\n        \n        // threads in bounds read input; others get neutral element\n        {\n            if (slt32(gtid_17840, sizze_17490)) {\n                int32_t x_17505 = ((__global\n                                    int32_t *) ",
                   "shp_mem_18302)[gtid_17840];\n                bool cond_17507 = gtid_17840 == 0;\n                int32_t res_17508;\n                \n                if (cond_17507) {\n                    res_17508 = 0;\n                } else {\n                    int32_t i_17509 = gtid_17840 - 1;\n                    int32_t res_17510 = ((__global\n                                          int32_t *) shp_mem_18302)[i_17509];\n                    \n                    res_17508 = res_17510;\n                }\n                // write to-scan values to parameters\n                {\n                    x_17501 = x_17505;\n                    x_17502 = res_17508;\n                }\n                // write mapped values results to global memory\n                { }\n            } else {\n                x_17501 = 0;\n                x_17502 = 0;\n            }\n        }\n        // combine with carry and write to local memory\n        {\n            int32_t res_17503 = x_17499 + x_17501;\n            int32_t res_17504 = x_17500 + x_17502;\n            \n            ((__local int32_t *) scan_arr_mem_18524)[local_tid_18520] =\n                res_17503;\n            ((__local int32_t *) scan_arr_mem_18526)[local_tid_18520] =\n                res_17504;\n        }\n        \n        int32_t x_18513;\n        int32_t x_18514;\n        int32_t x_18515;\n        int32_t x_18516;\n        int32_t x_18531;\n        int32_t x_18532;\n        int32_t x_18533;\n        int32_t x_18534;\n        int32_t skip_threads_18537;\n        \n        if (slt32(local_tid_18520, segscan_group_sizze_17836)) {\n            x_18515 = ((volatile __local\n                        int32_t *) scan_arr_mem_18524)[local_tid_18520];\n            x_18516 = ((volatile __local\n                        int32_t *) scan_arr_mem_18526)[local_tid_18520];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_18537 = 1;\n            while (slt32(skip_threads_18537, 32)) {\n                if (sle32(skip_threads_18537, local_t",
                   "id_18520 -\n                          squot32(local_tid_18520, 32) * 32) &&\n                    slt32(local_tid_18520, segscan_group_sizze_17836)) {\n                    // read operands\n                    {\n                        x_18513 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18524)[local_tid_18520 -\n                                                                   skip_threads_18537];\n                        x_18514 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18526)[local_tid_18520 -\n                                                                   skip_threads_18537];\n                    }\n                    // perform operation\n                    {\n                        int32_t res_18517 = x_18513 + x_18515;\n                        int32_t res_18518 = x_18514 + x_18516;\n                        \n                        x_18515 = res_18517;\n                        x_18516 = res_18518;\n                    }\n                }\n                if (sle32(wave_sizze_18522, skip_threads_18537)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_18537, local_tid_18520 -\n                          squot32(local_tid_18520, 32) * 32) &&\n                    slt32(local_tid_18520, segscan_group_sizze_17836)) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18524)[local_tid_18520] =\n                            x_18515;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18526)[local_tid_18520] =\n                            x_18516;\n                    }\n                }\n                if (sle32(wave_sizze_18522, skip_threads_18537)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_18537 *= 2;\n            }\n        }\n        barrier(CLK_LOCAL_MEM",
                   "_FENCE);\n        // last thread of block 'i' writes its result to offset 'i'\n        {\n            if ((local_tid_18520 - squot32(local_tid_18520, 32) * 32) == 31 &&\n                slt32(local_tid_18520, segscan_group_sizze_17836)) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18524)[squot32(local_tid_18520, 32)] =\n                    x_18515;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18526)[squot32(local_tid_18520, 32)] =\n                    x_18516;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n        {\n            int32_t skip_threads_18538;\n            \n            if (squot32(local_tid_18520, 32) == 0 && slt32(local_tid_18520,\n                                                           segscan_group_sizze_17836)) {\n                x_18533 = ((volatile __local\n                            int32_t *) scan_arr_mem_18524)[local_tid_18520];\n                x_18534 = ((volatile __local\n                            int32_t *) scan_arr_mem_18526)[local_tid_18520];\n            }\n            // in-block scan (hopefully no barriers needed)\n            {\n                skip_threads_18538 = 1;\n                while (slt32(skip_threads_18538, 32)) {\n                    if (sle32(skip_threads_18538, local_tid_18520 -\n                              squot32(local_tid_18520, 32) * 32) &&\n                        (squot32(local_tid_18520, 32) == 0 &&\n                         slt32(local_tid_18520, segscan_group_sizze_17836))) {\n                        // read operands\n                        {\n                            x_18531 = ((volatile __local\n                                        int32_t *) scan_arr_mem_18524)[local_tid_18520 -\n                                                                       skip_threads_18538];\n                            x_18532 = ((volatile __local\n                             ",
                   "           int32_t *) scan_arr_mem_18526)[local_tid_18520 -\n                                                                       skip_threads_18538];\n                        }\n                        // perform operation\n                        {\n                            int32_t res_18535 = x_18531 + x_18533;\n                            int32_t res_18536 = x_18532 + x_18534;\n                            \n                            x_18533 = res_18535;\n                            x_18534 = res_18536;\n                        }\n                    }\n                    if (sle32(wave_sizze_18522, skip_threads_18538)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    if (sle32(skip_threads_18538, local_tid_18520 -\n                              squot32(local_tid_18520, 32) * 32) &&\n                        (squot32(local_tid_18520, 32) == 0 &&\n                         slt32(local_tid_18520, segscan_group_sizze_17836))) {\n                        // write result\n                        {\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18524)[local_tid_18520] =\n                                x_18533;\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18526)[local_tid_18520] =\n                                x_18534;\n                        }\n                    }\n                    if (sle32(wave_sizze_18522, skip_threads_18538)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_18538 *= 2;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // carry-in for every block except the first\n        {\n            if (!(squot32(local_tid_18520, 32) == 0 || !slt32(local_tid_18520,\n                                                              segscan_group_sizze_17836))) {\n                // read operands\n                {\n          ",
                   "          x_18513 = ((volatile __local\n                                int32_t *) scan_arr_mem_18524)[squot32(local_tid_18520,\n                                                                       32) - 1];\n                    x_18514 = ((volatile __local\n                                int32_t *) scan_arr_mem_18526)[squot32(local_tid_18520,\n                                                                       32) - 1];\n                }\n                // perform operation\n                {\n                    int32_t res_18517 = x_18513 + x_18515;\n                    int32_t res_18518 = x_18514 + x_18516;\n                    \n                    x_18515 = res_18517;\n                    x_18516 = res_18518;\n                }\n                // write final result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18524)[local_tid_18520] = x_18515;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18526)[local_tid_18520] = x_18516;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // restore correct values for first block\n        {\n            if (squot32(local_tid_18520, 32) == 0) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18524)[local_tid_18520] = x_18515;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18526)[local_tid_18520] = x_18516;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // threads in bounds write partial scan result\n        {\n            if (slt32(gtid_17840, sizze_17490)) {\n                ((__global int32_t *) mem_18307)[gtid_17840] = ((__local\n                                                                 int32_t *) scan_arr_mem_18524)[local_tid_18520];\n                ((__global int32_t *) mem_18310)[gtid_17840] = ((__local\n                                                                 int32_t *) scan_arr_mem_18526)[local_tid_18520]",
                   ";\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // first thread reads last element as carry-in for next iteration\n        {\n            if (local_tid_18520 == 0) {\n                x_17499 = ((__local\n                            int32_t *) scan_arr_mem_18524)[segscan_group_sizze_17836 -\n                                                           1];\n                x_17500 = ((__local\n                            int32_t *) scan_arr_mem_18526)[segscan_group_sizze_17836 -\n                                                           1];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n}\n__kernel void scan_stage1_17862(__local volatile\n                                int64_t *scan_arr_mem_18596_backing_aligned_0,\n                                __local volatile\n                                int64_t *scan_arr_mem_18598_backing_aligned_1,\n                                int32_t aoa_len_17514, __global\n                                unsigned char *mem_18313, __global\n                                unsigned char *mem_18317, __global\n                                unsigned char *mem_18320,\n                                int32_t num_threads_18582)\n{\n    const int32_t segscan_group_sizze_17857 = mainzisegscan_group_sizze_17856;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_18596_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_18596_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_18598_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_18598_backing_aligned_1;\n    int32_t global_tid_18591;\n    int32_t local_tid_18592;\n    int32_t group_sizze_18595;\n    int32_t wave_sizze_18594;\n    int32_t group_tid_18593;\n    \n    global_tid_18591 = get_global_id(0);\n    local_tid_18592 = get_local_id(0);\n    group_sizze_18595 = ge",
                   "t_local_size(0);\n    wave_sizze_18594 = LOCKSTEP_WIDTH;\n    group_tid_18593 = get_group_id(0);\n    \n    int32_t phys_tid_17862 = global_tid_18591;\n    __local char *scan_arr_mem_18596;\n    \n    scan_arr_mem_18596 = (__local char *) scan_arr_mem_18596_backing_0;\n    \n    __local char *scan_arr_mem_18598;\n    \n    scan_arr_mem_18598 = (__local char *) scan_arr_mem_18598_backing_1;\n    \n    int32_t x_17525;\n    int32_t x_17526;\n    int32_t x_17527;\n    int32_t x_17528;\n    \n    x_17525 = 0;\n    x_17526 = 0;\n    for (int32_t j_18600 = 0; j_18600 < squot32(aoa_len_17514 +\n                                                num_threads_18582 - 1,\n                                                num_threads_18582); j_18600++) {\n        int32_t chunk_offset_18601 = segscan_group_sizze_17857 * j_18600 +\n                group_tid_18593 * (segscan_group_sizze_17857 *\n                                   squot32(aoa_len_17514 + num_threads_18582 -\n                                           1, num_threads_18582));\n        int32_t flat_idx_18602 = chunk_offset_18601 + local_tid_18592;\n        int32_t gtid_17861 = flat_idx_18602;\n        \n        // threads in bounds read input; others get neutral element\n        {\n            if (slt32(gtid_17861, aoa_len_17514)) {\n                int32_t x_17533 = ((__global int32_t *) mem_18313)[gtid_17861];\n                bool cond_17534 = x_17533 == 0;\n                int32_t res_17535;\n                \n                if (cond_17534) {\n                    res_17535 = 0;\n                } else {\n                    int32_t res_17536 = x_17533 - 1;\n                    \n                    res_17535 = res_17536;\n                }\n                // write to-scan values to parameters\n                {\n                    x_17527 = x_17533;\n                    x_17528 = res_17535;\n                }\n                // write mapped values results to global memory\n                { }\n            } else {\n                x_17527 = 0;\n                x_1752",
                   "8 = 0;\n            }\n        }\n        // combine with carry and write to local memory\n        {\n            int32_t f_17529 = x_17525 | x_17527;\n            bool cond_17530 = slt32(0, x_17527);\n            int32_t res_17531;\n            \n            if (cond_17530) {\n                res_17531 = x_17528;\n            } else {\n                int32_t res_17532 = x_17526 + x_17528;\n                \n                res_17531 = res_17532;\n            }\n            ((__local int32_t *) scan_arr_mem_18596)[local_tid_18592] = f_17529;\n            ((__local int32_t *) scan_arr_mem_18598)[local_tid_18592] =\n                res_17531;\n        }\n        \n        int32_t x_18583;\n        int32_t x_18584;\n        int32_t x_18585;\n        int32_t x_18586;\n        int32_t x_18603;\n        int32_t x_18604;\n        int32_t x_18605;\n        int32_t x_18606;\n        int32_t skip_threads_18611;\n        \n        if (slt32(local_tid_18592, segscan_group_sizze_17857)) {\n            x_18585 = ((volatile __local\n                        int32_t *) scan_arr_mem_18596)[local_tid_18592];\n            x_18586 = ((volatile __local\n                        int32_t *) scan_arr_mem_18598)[local_tid_18592];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_18611 = 1;\n            while (slt32(skip_threads_18611, 32)) {\n                if (sle32(skip_threads_18611, local_tid_18592 -\n                          squot32(local_tid_18592, 32) * 32) &&\n                    slt32(local_tid_18592, segscan_group_sizze_17857)) {\n                    // read operands\n                    {\n                        x_18583 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18596)[local_tid_18592 -\n                                                                   skip_threads_18611];\n                        x_18584 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18598)[local_tid_18592 -\n                       ",
                   "                                            skip_threads_18611];\n                    }\n                    // perform operation\n                    {\n                        int32_t f_18587 = x_18583 | x_18585;\n                        bool cond_18588 = slt32(0, x_18585);\n                        int32_t res_18589;\n                        \n                        if (cond_18588) {\n                            res_18589 = x_18586;\n                        } else {\n                            int32_t res_18590 = x_18584 + x_18586;\n                            \n                            res_18589 = res_18590;\n                        }\n                        x_18585 = f_18587;\n                        x_18586 = res_18589;\n                    }\n                }\n                if (sle32(wave_sizze_18594, skip_threads_18611)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_18611, local_tid_18592 -\n                          squot32(local_tid_18592, 32) * 32) &&\n                    slt32(local_tid_18592, segscan_group_sizze_17857)) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18596)[local_tid_18592] =\n                            x_18585;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18598)[local_tid_18592] =\n                            x_18586;\n                    }\n                }\n                if (sle32(wave_sizze_18594, skip_threads_18611)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_18611 *= 2;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // last thread of block 'i' writes its result to offset 'i'\n        {\n            if ((local_tid_18592 - squot32(local_tid_18592, 32) * 32) == 31 &&\n                slt32(local_tid_18592, segscan_group_sizze_17857)) {\n                ((volatil",
                   "e __local\n                  int32_t *) scan_arr_mem_18596)[squot32(local_tid_18592, 32)] =\n                    x_18585;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18598)[squot32(local_tid_18592, 32)] =\n                    x_18586;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n        {\n            int32_t skip_threads_18612;\n            \n            if (squot32(local_tid_18592, 32) == 0 && slt32(local_tid_18592,\n                                                           segscan_group_sizze_17857)) {\n                x_18605 = ((volatile __local\n                            int32_t *) scan_arr_mem_18596)[local_tid_18592];\n                x_18606 = ((volatile __local\n                            int32_t *) scan_arr_mem_18598)[local_tid_18592];\n            }\n            // in-block scan (hopefully no barriers needed)\n            {\n                skip_threads_18612 = 1;\n                while (slt32(skip_threads_18612, 32)) {\n                    if (sle32(skip_threads_18612, local_tid_18592 -\n                              squot32(local_tid_18592, 32) * 32) &&\n                        (squot32(local_tid_18592, 32) == 0 &&\n                         slt32(local_tid_18592, segscan_group_sizze_17857))) {\n                        // read operands\n                        {\n                            x_18603 = ((volatile __local\n                                        int32_t *) scan_arr_mem_18596)[local_tid_18592 -\n                                                                       skip_threads_18612];\n                            x_18604 = ((volatile __local\n                                        int32_t *) scan_arr_mem_18598)[local_tid_18592 -\n                                                                       skip_threads_18612];\n                        }\n                        // perform operation\n                        {\n           ",
                   "                 int32_t f_18607 = x_18603 | x_18605;\n                            bool cond_18608 = slt32(0, x_18605);\n                            int32_t res_18609;\n                            \n                            if (cond_18608) {\n                                res_18609 = x_18606;\n                            } else {\n                                int32_t res_18610 = x_18604 + x_18606;\n                                \n                                res_18609 = res_18610;\n                            }\n                            x_18605 = f_18607;\n                            x_18606 = res_18609;\n                        }\n                    }\n                    if (sle32(wave_sizze_18594, skip_threads_18612)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    if (sle32(skip_threads_18612, local_tid_18592 -\n                              squot32(local_tid_18592, 32) * 32) &&\n                        (squot32(local_tid_18592, 32) == 0 &&\n                         slt32(local_tid_18592, segscan_group_sizze_17857))) {\n                        // write result\n                        {\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18596)[local_tid_18592] =\n                                x_18605;\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18598)[local_tid_18592] =\n                                x_18606;\n                        }\n                    }\n                    if (sle32(wave_sizze_18594, skip_threads_18612)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_18612 *= 2;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // carry-in for every block except the first\n        {\n            if (!(squot32(local_tid_18592, 32) == 0 || !slt32(local_tid_18592,\n                                                ",
                   "              segscan_group_sizze_17857))) {\n                // read operands\n                {\n                    x_18583 = ((volatile __local\n                                int32_t *) scan_arr_mem_18596)[squot32(local_tid_18592,\n                                                                       32) - 1];\n                    x_18584 = ((volatile __local\n                                int32_t *) scan_arr_mem_18598)[squot32(local_tid_18592,\n                                                                       32) - 1];\n                }\n                // perform operation\n                {\n                    int32_t f_18587 = x_18583 | x_18585;\n                    bool cond_18588 = slt32(0, x_18585);\n                    int32_t res_18589;\n                    \n                    if (cond_18588) {\n                        res_18589 = x_18586;\n                    } else {\n                        int32_t res_18590 = x_18584 + x_18586;\n                        \n                        res_18589 = res_18590;\n                    }\n                    x_18585 = f_18587;\n                    x_18586 = res_18589;\n                }\n                // write final result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18596)[local_tid_18592] = x_18585;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18598)[local_tid_18592] = x_18586;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // restore correct values for first block\n        {\n            if (squot32(local_tid_18592, 32) == 0) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18596)[local_tid_18592] = x_18585;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18598)[local_tid_18592] = x_18586;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // threads in bounds write partial scan result\n        {\n            if",
                   " (slt32(gtid_17861, aoa_len_17514)) {\n                ((__global int32_t *) mem_18317)[gtid_17861] = ((__local\n                                                                 int32_t *) scan_arr_mem_18596)[local_tid_18592];\n                ((__global int32_t *) mem_18320)[gtid_17861] = ((__local\n                                                                 int32_t *) scan_arr_mem_18598)[local_tid_18592];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // first thread reads last element as carry-in for next iteration\n        {\n            if (local_tid_18592 == 0) {\n                x_17525 = ((__local\n                            int32_t *) scan_arr_mem_18596)[segscan_group_sizze_17857 -\n                                                           1];\n                x_17526 = ((__local\n                            int32_t *) scan_arr_mem_18598)[segscan_group_sizze_17857 -\n                                                           1];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n}\n__kernel void scan_stage1_17871(__local volatile\n                                int64_t *scan_arr_mem_18666_backing_aligned_0,\n                                int32_t sizze_17490, int32_t count_17495,\n                                __global unsigned char *shp_mem_18302, __global\n                                unsigned char *arr_mem_18303, __global\n                                unsigned char *mem_18307, __global\n                                unsigned char *mem_18324, __global\n                                unsigned char *mem_18327,\n                                int32_t num_threads_18657)\n{\n    const int32_t segscan_group_sizze_17866 = mainzisegscan_group_sizze_17865;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_18666_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_18666_backing_aligned_0;\n ",
                   "   int32_t global_tid_18661;\n    int32_t local_tid_18662;\n    int32_t group_sizze_18665;\n    int32_t wave_sizze_18664;\n    int32_t group_tid_18663;\n    \n    global_tid_18661 = get_global_id(0);\n    local_tid_18662 = get_local_id(0);\n    group_sizze_18665 = get_local_size(0);\n    wave_sizze_18664 = LOCKSTEP_WIDTH;\n    group_tid_18663 = get_group_id(0);\n    \n    int32_t phys_tid_17871 = global_tid_18661;\n    __local char *scan_arr_mem_18666;\n    \n    scan_arr_mem_18666 = (__local char *) scan_arr_mem_18666_backing_0;\n    \n    int32_t x_17556;\n    int32_t x_17557;\n    \n    x_17556 = 0;\n    for (int32_t j_18668 = 0; j_18668 < squot32(sizze_17490 +\n                                                num_threads_18657 - 1,\n                                                num_threads_18657); j_18668++) {\n        int32_t chunk_offset_18669 = segscan_group_sizze_17866 * j_18668 +\n                group_tid_18663 * (segscan_group_sizze_17866 *\n                                   squot32(sizze_17490 + num_threads_18657 - 1,\n                                           num_threads_18657));\n        int32_t flat_idx_18670 = chunk_offset_18669 + local_tid_18662;\n        int32_t gtid_17870 = flat_idx_18670;\n        \n        // threads in bounds read input; others get neutral element\n        {\n            if (slt32(gtid_17870, sizze_17490)) {\n                int32_t x_17559 = ((__global\n                                    int32_t *) shp_mem_18302)[gtid_17870];\n                int32_t randomInd_arg_17561 = x_17559 - 1;\n                bool cond_17562 = slt32(randomInd_arg_17561, 0);\n                int32_t res_17563;\n                \n                if (cond_17562) {\n                    res_17563 = 0;\n                } else {\n                    int32_t x_17564 = count_17495 + randomInd_arg_17561;\n                    int32_t y_17565 = 1 + randomInd_arg_17561;\n                    int32_t x_17566 = smod32(x_17564, y_17565);\n                    \n                    res_17563 = x_17566;\n         ",
                   "       }\n                \n                bool cond_17567 = sle32(x_17559, 0);\n                float res_17568;\n                \n                if (cond_17567) {\n                    res_17568 = 0.0F;\n                } else {\n                    bool cond_17569 = slt32(0, gtid_17870);\n                    int32_t off_17570;\n                    \n                    if (cond_17569) {\n                        int32_t i_17571 = gtid_17870 - 1;\n                        int32_t res_17572 = ((__global\n                                              int32_t *) mem_18307)[i_17571];\n                        \n                        off_17570 = res_17572;\n                    } else {\n                        off_17570 = 0;\n                    }\n                    \n                    int32_t i_17573 = res_17563 + off_17570;\n                    float res_17574 = ((__global\n                                        float *) arr_mem_18303)[i_17573];\n                    \n                    res_17568 = res_17574;\n                }\n                \n                bool cond_17575 = gtid_17870 == 0;\n                int32_t res_17576;\n                \n                if (cond_17575) {\n                    res_17576 = 0;\n                } else {\n                    int32_t i_17577 = gtid_17870 - 1;\n                    int32_t res_17578 = ((__global\n                                          int32_t *) shp_mem_18302)[i_17577];\n                    \n                    res_17576 = res_17578;\n                }\n                // write to-scan values to parameters\n                {\n                    x_17557 = res_17576;\n                }\n                // write mapped values results to global memory\n                {\n                    ((__global float *) mem_18327)[gtid_17870] = res_17568;\n                }\n            } else {\n                x_17557 = 0;\n            }\n        }\n        // combine with carry and write to local memory\n        {\n            int32_t res_17558 = x_17556 + x_17557;",
                   "\n            \n            ((__local int32_t *) scan_arr_mem_18666)[local_tid_18662] =\n                res_17558;\n        }\n        \n        int32_t x_18658;\n        int32_t x_18659;\n        int32_t x_18671;\n        int32_t x_18672;\n        int32_t skip_threads_18674;\n        \n        if (slt32(local_tid_18662, segscan_group_sizze_17866)) {\n            x_18659 = ((volatile __local\n                        int32_t *) scan_arr_mem_18666)[local_tid_18662];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_18674 = 1;\n            while (slt32(skip_threads_18674, 32)) {\n                if (sle32(skip_threads_18674, local_tid_18662 -\n                          squot32(local_tid_18662, 32) * 32) &&\n                    slt32(local_tid_18662, segscan_group_sizze_17866)) {\n                    // read operands\n                    {\n                        x_18658 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18666)[local_tid_18662 -\n                                                                   skip_threads_18674];\n                    }\n                    // perform operation\n                    {\n                        int32_t res_18660 = x_18658 + x_18659;\n                        \n                        x_18659 = res_18660;\n                    }\n                }\n                if (sle32(wave_sizze_18664, skip_threads_18674)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_18674, local_tid_18662 -\n                          squot32(local_tid_18662, 32) * 32) &&\n                    slt32(local_tid_18662, segscan_group_sizze_17866)) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18666)[local_tid_18662] =\n                            x_18659;\n                    }\n                }\n                if (sle32(wave_sizze_18664, sk",
                   "ip_threads_18674)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_18674 *= 2;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // last thread of block 'i' writes its result to offset 'i'\n        {\n            if ((local_tid_18662 - squot32(local_tid_18662, 32) * 32) == 31 &&\n                slt32(local_tid_18662, segscan_group_sizze_17866)) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18666)[squot32(local_tid_18662, 32)] =\n                    x_18659;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n        {\n            int32_t skip_threads_18675;\n            \n            if (squot32(local_tid_18662, 32) == 0 && slt32(local_tid_18662,\n                                                           segscan_group_sizze_17866)) {\n                x_18672 = ((volatile __local\n                            int32_t *) scan_arr_mem_18666)[local_tid_18662];\n            }\n            // in-block scan (hopefully no barriers needed)\n            {\n                skip_threads_18675 = 1;\n                while (slt32(skip_threads_18675, 32)) {\n                    if (sle32(skip_threads_18675, local_tid_18662 -\n                              squot32(local_tid_18662, 32) * 32) &&\n                        (squot32(local_tid_18662, 32) == 0 &&\n                         slt32(local_tid_18662, segscan_group_sizze_17866))) {\n                        // read operands\n                        {\n                            x_18671 = ((volatile __local\n                                        int32_t *) scan_arr_mem_18666)[local_tid_18662 -\n                                                                       skip_threads_18675];\n                        }\n                        // perform operation\n                        {\n                            int32_t res_18673 = x_18671 + x_18672;\n      ",
                   "                      \n                            x_18672 = res_18673;\n                        }\n                    }\n                    if (sle32(wave_sizze_18664, skip_threads_18675)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    if (sle32(skip_threads_18675, local_tid_18662 -\n                              squot32(local_tid_18662, 32) * 32) &&\n                        (squot32(local_tid_18662, 32) == 0 &&\n                         slt32(local_tid_18662, segscan_group_sizze_17866))) {\n                        // write result\n                        {\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18666)[local_tid_18662] =\n                                x_18672;\n                        }\n                    }\n                    if (sle32(wave_sizze_18664, skip_threads_18675)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_18675 *= 2;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // carry-in for every block except the first\n        {\n            if (!(squot32(local_tid_18662, 32) == 0 || !slt32(local_tid_18662,\n                                                              segscan_group_sizze_17866))) {\n                // read operands\n                {\n                    x_18658 = ((volatile __local\n                                int32_t *) scan_arr_mem_18666)[squot32(local_tid_18662,\n                                                                       32) - 1];\n                }\n                // perform operation\n                {\n                    int32_t res_18660 = x_18658 + x_18659;\n                    \n                    x_18659 = res_18660;\n                }\n                // write final result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18666)[local_tid_18662] = x_18659;\n          ",
                   "      }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // restore correct values for first block\n        {\n            if (squot32(local_tid_18662, 32) == 0) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18666)[local_tid_18662] = x_18659;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // threads in bounds write partial scan result\n        {\n            if (slt32(gtid_17870, sizze_17490)) {\n                ((__global int32_t *) mem_18324)[gtid_17870] = ((__local\n                                                                 int32_t *) scan_arr_mem_18666)[local_tid_18662];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // first thread reads last element as carry-in for next iteration\n        {\n            if (local_tid_18662 == 0) {\n                x_17556 = ((__local\n                            int32_t *) scan_arr_mem_18666)[segscan_group_sizze_17866 -\n                                                           1];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n}\n__kernel void scan_stage1_17960(__local volatile\n                                int64_t *scan_arr_mem_18743_backing_aligned_0,\n                                __local volatile\n                                int64_t *scan_arr_mem_18745_backing_aligned_1,\n                                __local volatile\n                                int64_t *scan_arr_mem_18747_backing_aligned_2,\n                                __local volatile\n                                int64_t *scan_arr_mem_18749_backing_aligned_3,\n                                __local volatile\n                                int64_t *scan_arr_mem_18751_backing_aligned_4,\n                                __local volatile\n                                int64_t *scan_arr_mem_18753_backing_aligned_5,\n                                int32_t aoa_len_17582, __global\n                                unsigned char *mem_18330, __global\n",
                   "                                unsigned char *mem_18333, __global\n                                unsigned char *mem_18339, __global\n                                unsigned char *mem_18342, __global\n                                unsigned char *mem_18345, __global\n                                unsigned char *mem_18348, __global\n                                unsigned char *mem_18351, __global\n                                unsigned char *mem_18354,\n                                int32_t num_threads_18713)\n{\n    const int32_t segscan_group_sizze_17955 = mainzisegscan_group_sizze_17954;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_18743_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_18743_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_18745_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_18745_backing_aligned_1;\n    __local volatile char *restrict scan_arr_mem_18747_backing_2 =\n                          (__local volatile\n                           char *) scan_arr_mem_18747_backing_aligned_2;\n    __local volatile char *restrict scan_arr_mem_18749_backing_3 =\n                          (__local volatile\n                           char *) scan_arr_mem_18749_backing_aligned_3;\n    __local volatile char *restrict scan_arr_mem_18751_backing_4 =\n                          (__local volatile\n                           char *) scan_arr_mem_18751_backing_aligned_4;\n    __local volatile char *restrict scan_arr_mem_18753_backing_5 =\n                          (__local volatile\n                           char *) scan_arr_mem_18753_backing_aligned_5;\n    int32_t global_tid_18738;\n    int32_t local_tid_18739;\n    int32_t group_sizze_18742;\n    int32_t wave_sizze_18741;\n    int32_t group_tid_18740;\n    \n    global_tid_18738 = get_global_id(0);\n    local_t",
                   "id_18739 = get_local_id(0);\n    group_sizze_18742 = get_local_size(0);\n    wave_sizze_18741 = LOCKSTEP_WIDTH;\n    group_tid_18740 = get_group_id(0);\n    \n    int32_t phys_tid_17960 = global_tid_18738;\n    __local char *scan_arr_mem_18743;\n    \n    scan_arr_mem_18743 = (__local char *) scan_arr_mem_18743_backing_0;\n    \n    __local char *scan_arr_mem_18745;\n    \n    scan_arr_mem_18745 = (__local char *) scan_arr_mem_18745_backing_1;\n    \n    __local char *scan_arr_mem_18747;\n    \n    scan_arr_mem_18747 = (__local char *) scan_arr_mem_18747_backing_2;\n    \n    __local char *scan_arr_mem_18749;\n    \n    scan_arr_mem_18749 = (__local char *) scan_arr_mem_18749_backing_3;\n    \n    __local char *scan_arr_mem_18751;\n    \n    scan_arr_mem_18751 = (__local char *) scan_arr_mem_18751_backing_4;\n    \n    __local char *scan_arr_mem_18753;\n    \n    scan_arr_mem_18753 = (__local char *) scan_arr_mem_18753_backing_5;\n    \n    int32_t x_17621;\n    int32_t x_17622;\n    int32_t x_17623;\n    int32_t x_17624;\n    int32_t x_17625;\n    int32_t x_17626;\n    int32_t x_17627;\n    int32_t x_17628;\n    int32_t x_17629;\n    int32_t x_17630;\n    int32_t x_17631;\n    int32_t x_17632;\n    \n    x_17621 = 0;\n    x_17622 = 0;\n    x_17623 = 0;\n    x_17624 = 0;\n    x_17625 = 0;\n    x_17626 = 0;\n    for (int32_t j_18755 = 0; j_18755 < squot32(aoa_len_17582 +\n                                                num_threads_18713 - 1,\n                                                num_threads_18713); j_18755++) {\n        int32_t chunk_offset_18756 = segscan_group_sizze_17955 * j_18755 +\n                group_tid_18740 * (segscan_group_sizze_17955 *\n                                   squot32(aoa_len_17582 + num_threads_18713 -\n                                           1, num_threads_18713));\n        int32_t flat_idx_18757 = chunk_offset_18756 + local_tid_18739;\n        int32_t gtid_17959 = flat_idx_18757;\n        \n        // threads in bounds read input; others get neutral element\n        {\n            if (s",
                   "lt32(gtid_17959, aoa_len_17582)) {\n                int32_t x_17645 = ((__global int32_t *) mem_18330)[gtid_17959];\n                int32_t x_17646 = ((__global int32_t *) mem_18333)[gtid_17959];\n                bool cond_17647 = x_17645 == 0;\n                int32_t res_17648;\n                \n                if (cond_17647) {\n                    res_17648 = 0;\n                } else {\n                    int32_t res_17649 = x_17645 - 1;\n                    \n                    res_17648 = res_17649;\n                }\n                \n                int32_t res_17650 = 1 - x_17646;\n                \n                // write to-scan values to parameters\n                {\n                    x_17627 = x_17645;\n                    x_17628 = res_17648;\n                    x_17629 = x_17645;\n                    x_17630 = x_17646;\n                    x_17631 = x_17645;\n                    x_17632 = res_17650;\n                }\n                // write mapped values results to global memory\n                { }\n            } else {\n                x_17627 = 0;\n                x_17628 = 0;\n                x_17629 = 0;\n                x_17630 = 0;\n                x_17631 = 0;\n                x_17632 = 0;\n            }\n        }\n        // combine with carry and write to local memory\n        {\n            int32_t f_17633 = x_17621 | x_17627;\n            bool cond_17634 = slt32(0, x_17627);\n            int32_t res_17635;\n            \n            if (cond_17634) {\n                res_17635 = x_17628;\n            } else {\n                int32_t res_17636 = x_17622 + x_17628;\n                \n                res_17635 = res_17636;\n            }\n            \n            int32_t f_17637 = x_17623 | x_17629;\n            bool cond_17638 = slt32(0, x_17629);\n            int32_t res_17639;\n            \n            if (cond_17638) {\n                res_17639 = x_17630;\n            } else {\n                int32_t res_17640 = x_17624 + x_17630;\n                \n                res_17639 ",
                   "= res_17640;\n            }\n            \n            int32_t f_17641 = x_17625 | x_17631;\n            bool cond_17642 = slt32(0, x_17631);\n            int32_t res_17643;\n            \n            if (cond_17642) {\n                res_17643 = x_17632;\n            } else {\n                int32_t res_17644 = x_17626 + x_17632;\n                \n                res_17643 = res_17644;\n            }\n            ((__local int32_t *) scan_arr_mem_18743)[local_tid_18739] = f_17633;\n            ((__local int32_t *) scan_arr_mem_18745)[local_tid_18739] =\n                res_17635;\n            ((__local int32_t *) scan_arr_mem_18747)[local_tid_18739] = f_17637;\n            ((__local int32_t *) scan_arr_mem_18749)[local_tid_18739] =\n                res_17639;\n            ((__local int32_t *) scan_arr_mem_18751)[local_tid_18739] = f_17641;\n            ((__local int32_t *) scan_arr_mem_18753)[local_tid_18739] =\n                res_17643;\n        }\n        \n        int32_t x_18714;\n        int32_t x_18715;\n        int32_t x_18716;\n        int32_t x_18717;\n        int32_t x_18718;\n        int32_t x_18719;\n        int32_t x_18720;\n        int32_t x_18721;\n        int32_t x_18722;\n        int32_t x_18723;\n        int32_t x_18724;\n        int32_t x_18725;\n        int32_t x_18758;\n        int32_t x_18759;\n        int32_t x_18760;\n        int32_t x_18761;\n        int32_t x_18762;\n        int32_t x_18763;\n        int32_t x_18764;\n        int32_t x_18765;\n        int32_t x_18766;\n        int32_t x_18767;\n        int32_t x_18768;\n        int32_t x_18769;\n        int32_t skip_threads_18782;\n        \n        if (slt32(local_tid_18739, segscan_group_sizze_17955)) {\n            x_18720 = ((volatile __local\n                        int32_t *) scan_arr_mem_18743)[local_tid_18739];\n            x_18721 = ((volatile __local\n                        int32_t *) scan_arr_mem_18745)[local_tid_18739];\n            x_18722 = ((volatile __local\n                        int32_t *) scan_arr_mem_18747)[local_tid_18",
                   "739];\n            x_18723 = ((volatile __local\n                        int32_t *) scan_arr_mem_18749)[local_tid_18739];\n            x_18724 = ((volatile __local\n                        int32_t *) scan_arr_mem_18751)[local_tid_18739];\n            x_18725 = ((volatile __local\n                        int32_t *) scan_arr_mem_18753)[local_tid_18739];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_18782 = 1;\n            while (slt32(skip_threads_18782, 32)) {\n                if (sle32(skip_threads_18782, local_tid_18739 -\n                          squot32(local_tid_18739, 32) * 32) &&\n                    slt32(local_tid_18739, segscan_group_sizze_17955)) {\n                    // read operands\n                    {\n                        x_18714 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18743)[local_tid_18739 -\n                                                                   skip_threads_18782];\n                        x_18715 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18745)[local_tid_18739 -\n                                                                   skip_threads_18782];\n                        x_18716 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18747)[local_tid_18739 -\n                                                                   skip_threads_18782];\n                        x_18717 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18749)[local_tid_18739 -\n                                                                   skip_threads_18782];\n                        x_18718 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18751)[local_tid_18739 -\n                                                                   skip_threads_18782];\n                        x_18719 = ((volatile __local\n                                    int32_t *",
                   ") scan_arr_mem_18753)[local_tid_18739 -\n                                                                   skip_threads_18782];\n                    }\n                    // perform operation\n                    {\n                        int32_t f_18726 = x_18714 | x_18720;\n                        bool cond_18727 = slt32(0, x_18720);\n                        int32_t res_18728;\n                        \n                        if (cond_18727) {\n                            res_18728 = x_18721;\n                        } else {\n                            int32_t res_18729 = x_18715 + x_18721;\n                            \n                            res_18728 = res_18729;\n                        }\n                        \n                        int32_t f_18730 = x_18716 | x_18722;\n                        bool cond_18731 = slt32(0, x_18722);\n                        int32_t res_18732;\n                        \n                        if (cond_18731) {\n                            res_18732 = x_18723;\n                        } else {\n                            int32_t res_18733 = x_18717 + x_18723;\n                            \n                            res_18732 = res_18733;\n                        }\n                        \n                        int32_t f_18734 = x_18718 | x_18724;\n                        bool cond_18735 = slt32(0, x_18724);\n                        int32_t res_18736;\n                        \n                        if (cond_18735) {\n                            res_18736 = x_18725;\n                        } else {\n                            int32_t res_18737 = x_18719 + x_18725;\n                            \n                            res_18736 = res_18737;\n                        }\n                        x_18720 = f_18726;\n                        x_18721 = res_18728;\n                        x_18722 = f_18730;\n                        x_18723 = res_18732;\n                        x_18724 = f_18734;\n                        x_18725 = res_18736;\n            ",
                   "        }\n                }\n                if (sle32(wave_sizze_18741, skip_threads_18782)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_18782, local_tid_18739 -\n                          squot32(local_tid_18739, 32) * 32) &&\n                    slt32(local_tid_18739, segscan_group_sizze_17955)) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18743)[local_tid_18739] =\n                            x_18720;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18745)[local_tid_18739] =\n                            x_18721;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18747)[local_tid_18739] =\n                            x_18722;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18749)[local_tid_18739] =\n                            x_18723;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18751)[local_tid_18739] =\n                            x_18724;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18753)[local_tid_18739] =\n                            x_18725;\n                    }\n                }\n                if (sle32(wave_sizze_18741, skip_threads_18782)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_18782 *= 2;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // last thread of block 'i' writes its result to offset 'i'\n        {\n            if ((local_tid_18739 - squot32(local_tid_18739, 32) * 32) == 31 &&\n                slt32(local_tid_18739, segscan_group_sizze_17955)) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18743)[squot32(local_tid_18739, 32)] =\n                    x_",
                   "18720;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18745)[squot32(local_tid_18739, 32)] =\n                    x_18721;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18747)[squot32(local_tid_18739, 32)] =\n                    x_18722;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18749)[squot32(local_tid_18739, 32)] =\n                    x_18723;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18751)[squot32(local_tid_18739, 32)] =\n                    x_18724;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18753)[squot32(local_tid_18739, 32)] =\n                    x_18725;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n        {\n            int32_t skip_threads_18783;\n            \n            if (squot32(local_tid_18739, 32) == 0 && slt32(local_tid_18739,\n                                                           segscan_group_sizze_17955)) {\n                x_18764 = ((volatile __local\n                            int32_t *) scan_arr_mem_18743)[local_tid_18739];\n                x_18765 = ((volatile __local\n                            int32_t *) scan_arr_mem_18745)[local_tid_18739];\n                x_18766 = ((volatile __local\n                            int32_t *) scan_arr_mem_18747)[local_tid_18739];\n                x_18767 = ((volatile __local\n                            int32_t *) scan_arr_mem_18749)[local_tid_18739];\n                x_18768 = ((volatile __local\n                            int32_t *) scan_arr_mem_18751)[local_tid_18739];\n                x_18769 = ((volatile __local\n                            int32_t *) scan_arr_mem_18753)[local_tid_18739];\n            }\n            // in-block scan (hopefully no barriers needed)\n            {\n                skip_threads_18783 = 1;\n                while (slt32(",
                   "skip_threads_18783, 32)) {\n                    if (sle32(skip_threads_18783, local_tid_18739 -\n                              squot32(local_tid_18739, 32) * 32) &&\n                        (squot32(local_tid_18739, 32) == 0 &&\n                         slt32(local_tid_18739, segscan_group_sizze_17955))) {\n                        // read operands\n                        {\n                            x_18758 = ((volatile __local\n                                        int32_t *) scan_arr_mem_18743)[local_tid_18739 -\n                                                                       skip_threads_18783];\n                            x_18759 = ((volatile __local\n                                        int32_t *) scan_arr_mem_18745)[local_tid_18739 -\n                                                                       skip_threads_18783];\n                            x_18760 = ((volatile __local\n                                        int32_t *) scan_arr_mem_18747)[local_tid_18739 -\n                                                                       skip_threads_18783];\n                            x_18761 = ((volatile __local\n                                        int32_t *) scan_arr_mem_18749)[local_tid_18739 -\n                                                                       skip_threads_18783];\n                            x_18762 = ((volatile __local\n                                        int32_t *) scan_arr_mem_18751)[local_tid_18739 -\n                                                                       skip_threads_18783];\n                            x_18763 = ((volatile __local\n                                        int32_t *) scan_arr_mem_18753)[local_tid_18739 -\n                                                                       skip_threads_18783];\n                        }\n                        // perform operation\n                        {\n                            int32_t f_18770 = x_18758 | x_18764;\n                            bool cond_1",
                   "8771 = slt32(0, x_18764);\n                            int32_t res_18772;\n                            \n                            if (cond_18771) {\n                                res_18772 = x_18765;\n                            } else {\n                                int32_t res_18773 = x_18759 + x_18765;\n                                \n                                res_18772 = res_18773;\n                            }\n                            \n                            int32_t f_18774 = x_18760 | x_18766;\n                            bool cond_18775 = slt32(0, x_18766);\n                            int32_t res_18776;\n                            \n                            if (cond_18775) {\n                                res_18776 = x_18767;\n                            } else {\n                                int32_t res_18777 = x_18761 + x_18767;\n                                \n                                res_18776 = res_18777;\n                            }\n                            \n                            int32_t f_18778 = x_18762 | x_18768;\n                            bool cond_18779 = slt32(0, x_18768);\n                            int32_t res_18780;\n                            \n                            if (cond_18779) {\n                                res_18780 = x_18769;\n                            } else {\n                                int32_t res_18781 = x_18763 + x_18769;\n                                \n                                res_18780 = res_18781;\n                            }\n                            x_18764 = f_18770;\n                            x_18765 = res_18772;\n                            x_18766 = f_18774;\n                            x_18767 = res_18776;\n                            x_18768 = f_18778;\n                            x_18769 = res_18780;\n                        }\n                    }\n                    if (sle32(wave_sizze_18741, skip_threads_18783)) {\n                        barrier(CLK_LOCAL_MEM_",
                   "FENCE);\n                    }\n                    if (sle32(skip_threads_18783, local_tid_18739 -\n                              squot32(local_tid_18739, 32) * 32) &&\n                        (squot32(local_tid_18739, 32) == 0 &&\n                         slt32(local_tid_18739, segscan_group_sizze_17955))) {\n                        // write result\n                        {\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18743)[local_tid_18739] =\n                                x_18764;\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18745)[local_tid_18739] =\n                                x_18765;\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18747)[local_tid_18739] =\n                                x_18766;\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18749)[local_tid_18739] =\n                                x_18767;\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18751)[local_tid_18739] =\n                                x_18768;\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18753)[local_tid_18739] =\n                                x_18769;\n                        }\n                    }\n                    if (sle32(wave_sizze_18741, skip_threads_18783)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_18783 *= 2;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // carry-in for every block except the first\n        {\n            if (!(squot32(local_tid_18739, 32) == 0 || !slt32(local_tid_18739,\n                                                              segscan_group_sizze_17955))) {\n                // read operands\n                {\n             ",
                   "       x_18714 = ((volatile __local\n                                int32_t *) scan_arr_mem_18743)[squot32(local_tid_18739,\n                                                                       32) - 1];\n                    x_18715 = ((volatile __local\n                                int32_t *) scan_arr_mem_18745)[squot32(local_tid_18739,\n                                                                       32) - 1];\n                    x_18716 = ((volatile __local\n                                int32_t *) scan_arr_mem_18747)[squot32(local_tid_18739,\n                                                                       32) - 1];\n                    x_18717 = ((volatile __local\n                                int32_t *) scan_arr_mem_18749)[squot32(local_tid_18739,\n                                                                       32) - 1];\n                    x_18718 = ((volatile __local\n                                int32_t *) scan_arr_mem_18751)[squot32(local_tid_18739,\n                                                                       32) - 1];\n                    x_18719 = ((volatile __local\n                                int32_t *) scan_arr_mem_18753)[squot32(local_tid_18739,\n                                                                       32) - 1];\n                }\n                // perform operation\n                {\n                    int32_t f_18726 = x_18714 | x_18720;\n                    bool cond_18727 = slt32(0, x_18720);\n                    int32_t res_18728;\n                    \n                    if (cond_18727) {\n                        res_18728 = x_18721;\n                    } else {\n                        int32_t res_18729 = x_18715 + x_18721;\n                        \n                        res_18728 = res_18729;\n                    }\n                    \n                    int32_t f_18730 = x_18716 | x_18722;\n                    bool cond_18731 = slt32(0, x_18722);\n                    int32_t res_18732;\n               ",
                   "     \n                    if (cond_18731) {\n                        res_18732 = x_18723;\n                    } else {\n                        int32_t res_18733 = x_18717 + x_18723;\n                        \n                        res_18732 = res_18733;\n                    }\n                    \n                    int32_t f_18734 = x_18718 | x_18724;\n                    bool cond_18735 = slt32(0, x_18724);\n                    int32_t res_18736;\n                    \n                    if (cond_18735) {\n                        res_18736 = x_18725;\n                    } else {\n                        int32_t res_18737 = x_18719 + x_18725;\n                        \n                        res_18736 = res_18737;\n                    }\n                    x_18720 = f_18726;\n                    x_18721 = res_18728;\n                    x_18722 = f_18730;\n                    x_18723 = res_18732;\n                    x_18724 = f_18734;\n                    x_18725 = res_18736;\n                }\n                // write final result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18743)[local_tid_18739] = x_18720;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18745)[local_tid_18739] = x_18721;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18747)[local_tid_18739] = x_18722;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18749)[local_tid_18739] = x_18723;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18751)[local_tid_18739] = x_18724;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18753)[local_tid_18739] = x_18725;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // restore correct values for first block\n        {\n            if (squot32(local_tid_18739, 32) == 0) {\n                ((volatile __local",
                   "\n                  int32_t *) scan_arr_mem_18743)[local_tid_18739] = x_18720;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18745)[local_tid_18739] = x_18721;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18747)[local_tid_18739] = x_18722;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18749)[local_tid_18739] = x_18723;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18751)[local_tid_18739] = x_18724;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18753)[local_tid_18739] = x_18725;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // threads in bounds write partial scan result\n        {\n            if (slt32(gtid_17959, aoa_len_17582)) {\n                ((__global int32_t *) mem_18339)[gtid_17959] = ((__local\n                                                                 int32_t *) scan_arr_mem_18743)[local_tid_18739];\n                ((__global int32_t *) mem_18342)[gtid_17959] = ((__local\n                                                                 int32_t *) scan_arr_mem_18745)[local_tid_18739];\n                ((__global int32_t *) mem_18345)[gtid_17959] = ((__local\n                                                                 int32_t *) scan_arr_mem_18747)[local_tid_18739];\n                ((__global int32_t *) mem_18348)[gtid_17959] = ((__local\n                                                                 int32_t *) scan_arr_mem_18749)[local_tid_18739];\n                ((__global int32_t *) mem_18351)[gtid_17959] = ((__local\n                                                                 int32_t *) scan_arr_mem_18751)[local_tid_18739];\n                ((__global int32_t *) mem_18354)[gtid_17959] = ((__local\n                                                                 int32_t *) scan_arr_mem_18753)[local_tid_18739];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);",
                   "\n        // first thread reads last element as carry-in for next iteration\n        {\n            if (local_tid_18739 == 0) {\n                x_17621 = ((__local\n                            int32_t *) scan_arr_mem_18743)[segscan_group_sizze_17955 -\n                                                           1];\n                x_17622 = ((__local\n                            int32_t *) scan_arr_mem_18745)[segscan_group_sizze_17955 -\n                                                           1];\n                x_17623 = ((__local\n                            int32_t *) scan_arr_mem_18747)[segscan_group_sizze_17955 -\n                                                           1];\n                x_17624 = ((__local\n                            int32_t *) scan_arr_mem_18749)[segscan_group_sizze_17955 -\n                                                           1];\n                x_17625 = ((__local\n                            int32_t *) scan_arr_mem_18751)[segscan_group_sizze_17955 -\n                                                           1];\n                x_17626 = ((__local\n                            int32_t *) scan_arr_mem_18753)[segscan_group_sizze_17955 -\n                                                           1];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n}\n__kernel void scan_stage1_17969(__local volatile\n                                int64_t *scan_arr_mem_18893_backing_aligned_0,\n                                int32_t sizze_17490, __global\n                                unsigned char *shp_mem_18302, __global\n                                unsigned char *mem_18358,\n                                int32_t num_threads_18884)\n{\n    const int32_t segscan_group_sizze_17964 = mainzisegscan_group_sizze_17963;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_18893_backing_0 =\n                          (__local volatile\n                           ch",
                   "ar *) scan_arr_mem_18893_backing_aligned_0;\n    int32_t global_tid_18888;\n    int32_t local_tid_18889;\n    int32_t group_sizze_18892;\n    int32_t wave_sizze_18891;\n    int32_t group_tid_18890;\n    \n    global_tid_18888 = get_global_id(0);\n    local_tid_18889 = get_local_id(0);\n    group_sizze_18892 = get_local_size(0);\n    wave_sizze_18891 = LOCKSTEP_WIDTH;\n    group_tid_18890 = get_group_id(0);\n    \n    int32_t phys_tid_17969 = global_tid_18888;\n    __local char *scan_arr_mem_18893;\n    \n    scan_arr_mem_18893 = (__local char *) scan_arr_mem_18893_backing_0;\n    \n    int32_t x_17663;\n    int32_t x_17664;\n    \n    x_17663 = 0;\n    for (int32_t j_18895 = 0; j_18895 < squot32(sizze_17490 +\n                                                num_threads_18884 - 1,\n                                                num_threads_18884); j_18895++) {\n        int32_t chunk_offset_18896 = segscan_group_sizze_17964 * j_18895 +\n                group_tid_18890 * (segscan_group_sizze_17964 *\n                                   squot32(sizze_17490 + num_threads_18884 - 1,\n                                           num_threads_18884));\n        int32_t flat_idx_18897 = chunk_offset_18896 + local_tid_18889;\n        int32_t gtid_17968 = flat_idx_18897;\n        \n        // threads in bounds read input; others get neutral element\n        {\n            if (slt32(gtid_17968, sizze_17490)) {\n                int32_t x_17666 = ((__global\n                                    int32_t *) shp_mem_18302)[gtid_17968];\n                \n                // write to-scan values to parameters\n                {\n                    x_17664 = x_17666;\n                }\n                // write mapped values results to global memory\n                { }\n            } else {\n                x_17664 = 0;\n            }\n        }\n        // combine with carry and write to local memory\n        {\n            int32_t res_17665 = x_17663 + x_17664;\n            \n            ((__local int32_t *) scan_arr_mem_18893)[local_tid",
                   "_18889] =\n                res_17665;\n        }\n        \n        int32_t x_18885;\n        int32_t x_18886;\n        int32_t x_18898;\n        int32_t x_18899;\n        int32_t skip_threads_18901;\n        \n        if (slt32(local_tid_18889, segscan_group_sizze_17964)) {\n            x_18886 = ((volatile __local\n                        int32_t *) scan_arr_mem_18893)[local_tid_18889];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_18901 = 1;\n            while (slt32(skip_threads_18901, 32)) {\n                if (sle32(skip_threads_18901, local_tid_18889 -\n                          squot32(local_tid_18889, 32) * 32) &&\n                    slt32(local_tid_18889, segscan_group_sizze_17964)) {\n                    // read operands\n                    {\n                        x_18885 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18893)[local_tid_18889 -\n                                                                   skip_threads_18901];\n                    }\n                    // perform operation\n                    {\n                        int32_t res_18887 = x_18885 + x_18886;\n                        \n                        x_18886 = res_18887;\n                    }\n                }\n                if (sle32(wave_sizze_18891, skip_threads_18901)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_18901, local_tid_18889 -\n                          squot32(local_tid_18889, 32) * 32) &&\n                    slt32(local_tid_18889, segscan_group_sizze_17964)) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18893)[local_tid_18889] =\n                            x_18886;\n                    }\n                }\n                if (sle32(wave_sizze_18891, skip_threads_18901)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n     ",
                   "           }\n                skip_threads_18901 *= 2;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // last thread of block 'i' writes its result to offset 'i'\n        {\n            if ((local_tid_18889 - squot32(local_tid_18889, 32) * 32) == 31 &&\n                slt32(local_tid_18889, segscan_group_sizze_17964)) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18893)[squot32(local_tid_18889, 32)] =\n                    x_18886;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n        {\n            int32_t skip_threads_18902;\n            \n            if (squot32(local_tid_18889, 32) == 0 && slt32(local_tid_18889,\n                                                           segscan_group_sizze_17964)) {\n                x_18899 = ((volatile __local\n                            int32_t *) scan_arr_mem_18893)[local_tid_18889];\n            }\n            // in-block scan (hopefully no barriers needed)\n            {\n                skip_threads_18902 = 1;\n                while (slt32(skip_threads_18902, 32)) {\n                    if (sle32(skip_threads_18902, local_tid_18889 -\n                              squot32(local_tid_18889, 32) * 32) &&\n                        (squot32(local_tid_18889, 32) == 0 &&\n                         slt32(local_tid_18889, segscan_group_sizze_17964))) {\n                        // read operands\n                        {\n                            x_18898 = ((volatile __local\n                                        int32_t *) scan_arr_mem_18893)[local_tid_18889 -\n                                                                       skip_threads_18902];\n                        }\n                        // perform operation\n                        {\n                            int32_t res_18900 = x_18898 + x_18899;\n                            \n                            x_18899 = res_18900;\n    ",
                   "                    }\n                    }\n                    if (sle32(wave_sizze_18891, skip_threads_18902)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    if (sle32(skip_threads_18902, local_tid_18889 -\n                              squot32(local_tid_18889, 32) * 32) &&\n                        (squot32(local_tid_18889, 32) == 0 &&\n                         slt32(local_tid_18889, segscan_group_sizze_17964))) {\n                        // write result\n                        {\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18893)[local_tid_18889] =\n                                x_18899;\n                        }\n                    }\n                    if (sle32(wave_sizze_18891, skip_threads_18902)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_18902 *= 2;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // carry-in for every block except the first\n        {\n            if (!(squot32(local_tid_18889, 32) == 0 || !slt32(local_tid_18889,\n                                                              segscan_group_sizze_17964))) {\n                // read operands\n                {\n                    x_18885 = ((volatile __local\n                                int32_t *) scan_arr_mem_18893)[squot32(local_tid_18889,\n                                                                       32) - 1];\n                }\n                // perform operation\n                {\n                    int32_t res_18887 = x_18885 + x_18886;\n                    \n                    x_18886 = res_18887;\n                }\n                // write final result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18893)[local_tid_18889] = x_18886;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n      ",
                   "  // restore correct values for first block\n        {\n            if (squot32(local_tid_18889, 32) == 0) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18893)[local_tid_18889] = x_18886;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // threads in bounds write partial scan result\n        {\n            if (slt32(gtid_17968, sizze_17490)) {\n                ((__global int32_t *) mem_18358)[gtid_17968] = ((__local\n                                                                 int32_t *) scan_arr_mem_18893)[local_tid_18889];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // first thread reads last element as carry-in for next iteration\n        {\n            if (local_tid_18889 == 0) {\n                x_17663 = ((__local\n                            int32_t *) scan_arr_mem_18893)[segscan_group_sizze_17964 -\n                                                           1];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n}\n__kernel void scan_stage1_18044(__local volatile\n                                int64_t *scan_arr_mem_18956_backing_aligned_0,\n                                __local volatile\n                                int64_t *scan_arr_mem_18958_backing_aligned_1,\n                                int32_t sizze_17490, __global\n                                unsigned char *shp_mem_18302, __global\n                                unsigned char *mem_18368, __global\n                                unsigned char *mem_18371,\n                                int32_t num_threads_18944)\n{\n    const int32_t segscan_group_sizze_18039 = mainzisegscan_group_sizze_18038;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_18956_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_18956_backing_aligned_0;\n    __local volatile char *restrict scan_arr_me",
                   "m_18958_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_18958_backing_aligned_1;\n    int32_t global_tid_18951;\n    int32_t local_tid_18952;\n    int32_t group_sizze_18955;\n    int32_t wave_sizze_18954;\n    int32_t group_tid_18953;\n    \n    global_tid_18951 = get_global_id(0);\n    local_tid_18952 = get_local_id(0);\n    group_sizze_18955 = get_local_size(0);\n    wave_sizze_18954 = LOCKSTEP_WIDTH;\n    group_tid_18953 = get_group_id(0);\n    \n    int32_t phys_tid_18044 = global_tid_18951;\n    __local char *scan_arr_mem_18956;\n    \n    scan_arr_mem_18956 = (__local char *) scan_arr_mem_18956_backing_0;\n    \n    __local char *scan_arr_mem_18958;\n    \n    scan_arr_mem_18958 = (__local char *) scan_arr_mem_18958_backing_1;\n    \n    int32_t x_17695;\n    int32_t x_17696;\n    int32_t x_17697;\n    int32_t x_17698;\n    \n    x_17695 = 0;\n    x_17696 = 0;\n    for (int32_t j_18960 = 0; j_18960 < squot32(sizze_17490 +\n                                                num_threads_18944 - 1,\n                                                num_threads_18944); j_18960++) {\n        int32_t chunk_offset_18961 = segscan_group_sizze_18039 * j_18960 +\n                group_tid_18953 * (segscan_group_sizze_18039 *\n                                   squot32(sizze_17490 + num_threads_18944 - 1,\n                                           num_threads_18944));\n        int32_t flat_idx_18962 = chunk_offset_18961 + local_tid_18952;\n        int32_t gtid_18043 = flat_idx_18962;\n        \n        // threads in bounds read input; others get neutral element\n        {\n            if (slt32(gtid_18043, sizze_17490)) {\n                bool cond_17702 = slt32(0, gtid_18043);\n                int32_t res_17703;\n                \n                if (cond_17702) {\n                    int32_t i_17704 = gtid_18043 - 1;\n                    int32_t res_17705 = ((__global\n                                          int32_t *) shp_mem_18302)[i_17704];\n                   ",
                   " \n                    res_17703 = res_17705;\n                } else {\n                    res_17703 = 0;\n                }\n                \n                bool cond_17706 = gtid_18043 == 0;\n                int32_t res_17707;\n                \n                if (cond_17706) {\n                    res_17707 = 0;\n                } else {\n                    int32_t i_17708 = gtid_18043 - 1;\n                    int32_t res_17709 = ((__global\n                                          int32_t *) shp_mem_18302)[i_17708];\n                    \n                    res_17707 = res_17709;\n                }\n                // write to-scan values to parameters\n                {\n                    x_17697 = res_17703;\n                    x_17698 = res_17707;\n                }\n                // write mapped values results to global memory\n                { }\n            } else {\n                x_17697 = 0;\n                x_17698 = 0;\n            }\n        }\n        // combine with carry and write to local memory\n        {\n            int32_t res_17699 = x_17695 + x_17697;\n            int32_t res_17700 = x_17696 + x_17698;\n            \n            ((__local int32_t *) scan_arr_mem_18956)[local_tid_18952] =\n                res_17699;\n            ((__local int32_t *) scan_arr_mem_18958)[local_tid_18952] =\n                res_17700;\n        }\n        \n        int32_t x_18945;\n        int32_t x_18946;\n        int32_t x_18947;\n        int32_t x_18948;\n        int32_t x_18963;\n        int32_t x_18964;\n        int32_t x_18965;\n        int32_t x_18966;\n        int32_t skip_threads_18969;\n        \n        if (slt32(local_tid_18952, segscan_group_sizze_18039)) {\n            x_18947 = ((volatile __local\n                        int32_t *) scan_arr_mem_18956)[local_tid_18952];\n            x_18948 = ((volatile __local\n                        int32_t *) scan_arr_mem_18958)[local_tid_18952];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_18",
                   "969 = 1;\n            while (slt32(skip_threads_18969, 32)) {\n                if (sle32(skip_threads_18969, local_tid_18952 -\n                          squot32(local_tid_18952, 32) * 32) &&\n                    slt32(local_tid_18952, segscan_group_sizze_18039)) {\n                    // read operands\n                    {\n                        x_18945 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18956)[local_tid_18952 -\n                                                                   skip_threads_18969];\n                        x_18946 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18958)[local_tid_18952 -\n                                                                   skip_threads_18969];\n                    }\n                    // perform operation\n                    {\n                        int32_t res_18949 = x_18945 + x_18947;\n                        int32_t res_18950 = x_18946 + x_18948;\n                        \n                        x_18947 = res_18949;\n                        x_18948 = res_18950;\n                    }\n                }\n                if (sle32(wave_sizze_18954, skip_threads_18969)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_18969, local_tid_18952 -\n                          squot32(local_tid_18952, 32) * 32) &&\n                    slt32(local_tid_18952, segscan_group_sizze_18039)) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18956)[local_tid_18952] =\n                            x_18947;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18958)[local_tid_18952] =\n                            x_18948;\n                    }\n                }\n                if (sle32(wave_sizze_18954, skip_threads_18969)) {\n                    barrier(CLK_LOCAL_MEM_FENCE)",
                   ";\n                }\n                skip_threads_18969 *= 2;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // last thread of block 'i' writes its result to offset 'i'\n        {\n            if ((local_tid_18952 - squot32(local_tid_18952, 32) * 32) == 31 &&\n                slt32(local_tid_18952, segscan_group_sizze_18039)) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18956)[squot32(local_tid_18952, 32)] =\n                    x_18947;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18958)[squot32(local_tid_18952, 32)] =\n                    x_18948;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n        {\n            int32_t skip_threads_18970;\n            \n            if (squot32(local_tid_18952, 32) == 0 && slt32(local_tid_18952,\n                                                           segscan_group_sizze_18039)) {\n                x_18965 = ((volatile __local\n                            int32_t *) scan_arr_mem_18956)[local_tid_18952];\n                x_18966 = ((volatile __local\n                            int32_t *) scan_arr_mem_18958)[local_tid_18952];\n            }\n            // in-block scan (hopefully no barriers needed)\n            {\n                skip_threads_18970 = 1;\n                while (slt32(skip_threads_18970, 32)) {\n                    if (sle32(skip_threads_18970, local_tid_18952 -\n                              squot32(local_tid_18952, 32) * 32) &&\n                        (squot32(local_tid_18952, 32) == 0 &&\n                         slt32(local_tid_18952, segscan_group_sizze_18039))) {\n                        // read operands\n                        {\n                            x_18963 = ((volatile __local\n                                        int32_t *) scan_arr_mem_18956)[local_tid_18952 -\n                                                                ",
                   "       skip_threads_18970];\n                            x_18964 = ((volatile __local\n                                        int32_t *) scan_arr_mem_18958)[local_tid_18952 -\n                                                                       skip_threads_18970];\n                        }\n                        // perform operation\n                        {\n                            int32_t res_18967 = x_18963 + x_18965;\n                            int32_t res_18968 = x_18964 + x_18966;\n                            \n                            x_18965 = res_18967;\n                            x_18966 = res_18968;\n                        }\n                    }\n                    if (sle32(wave_sizze_18954, skip_threads_18970)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    if (sle32(skip_threads_18970, local_tid_18952 -\n                              squot32(local_tid_18952, 32) * 32) &&\n                        (squot32(local_tid_18952, 32) == 0 &&\n                         slt32(local_tid_18952, segscan_group_sizze_18039))) {\n                        // write result\n                        {\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18956)[local_tid_18952] =\n                                x_18965;\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_18958)[local_tid_18952] =\n                                x_18966;\n                        }\n                    }\n                    if (sle32(wave_sizze_18954, skip_threads_18970)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_18970 *= 2;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // carry-in for every block except the first\n        {\n            if (!(squot32(local_tid_18952, 32) == 0 || !slt32(local_tid_18952,\n                                        ",
                   "                      segscan_group_sizze_18039))) {\n                // read operands\n                {\n                    x_18945 = ((volatile __local\n                                int32_t *) scan_arr_mem_18956)[squot32(local_tid_18952,\n                                                                       32) - 1];\n                    x_18946 = ((volatile __local\n                                int32_t *) scan_arr_mem_18958)[squot32(local_tid_18952,\n                                                                       32) - 1];\n                }\n                // perform operation\n                {\n                    int32_t res_18949 = x_18945 + x_18947;\n                    int32_t res_18950 = x_18946 + x_18948;\n                    \n                    x_18947 = res_18949;\n                    x_18948 = res_18950;\n                }\n                // write final result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18956)[local_tid_18952] = x_18947;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18958)[local_tid_18952] = x_18948;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // restore correct values for first block\n        {\n            if (squot32(local_tid_18952, 32) == 0) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18956)[local_tid_18952] = x_18947;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18958)[local_tid_18952] = x_18948;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // threads in bounds write partial scan result\n        {\n            if (slt32(gtid_18043, sizze_17490)) {\n                ((__global int32_t *) mem_18368)[gtid_18043] = ((__local\n                                                                 int32_t *) scan_arr_mem_18956)[local_tid_18952];\n                ((__global int32_t *) mem_18371)[gtid_18043] = ((__loca",
                   "l\n                                                                 int32_t *) scan_arr_mem_18958)[local_tid_18952];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // first thread reads last element as carry-in for next iteration\n        {\n            if (local_tid_18952 == 0) {\n                x_17695 = ((__local\n                            int32_t *) scan_arr_mem_18956)[segscan_group_sizze_18039 -\n                                                           1];\n                x_17696 = ((__local\n                            int32_t *) scan_arr_mem_18958)[segscan_group_sizze_18039 -\n                                                           1];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n}\n__kernel void scan_stage1_18065(__local volatile\n                                int64_t *scan_arr_mem_19029_backing_aligned_0,\n                                __local volatile\n                                int64_t *scan_arr_mem_19031_backing_aligned_1,\n                                int32_t aoa_len_17711, __global\n                                unsigned char *mem_18374, __global\n                                unsigned char *mem_18378, __global\n                                unsigned char *mem_18381,\n                                int32_t num_threads_19014)\n{\n    const int32_t segscan_group_sizze_18060 = mainzisegscan_group_sizze_18059;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_19029_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_19029_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_19031_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_19031_backing_aligned_1;\n    int32_t global_tid_19024;\n    int32_t local_tid_19025;\n    int32_t group_sizze_19028;\n    int32_t wave_sizze_19027;\n    int32_t group_tid_1902",
                   "6;\n    \n    global_tid_19024 = get_global_id(0);\n    local_tid_19025 = get_local_id(0);\n    group_sizze_19028 = get_local_size(0);\n    wave_sizze_19027 = LOCKSTEP_WIDTH;\n    group_tid_19026 = get_group_id(0);\n    \n    int32_t phys_tid_18065 = global_tid_19024;\n    __local char *scan_arr_mem_19029;\n    \n    scan_arr_mem_19029 = (__local char *) scan_arr_mem_19029_backing_0;\n    \n    __local char *scan_arr_mem_19031;\n    \n    scan_arr_mem_19031 = (__local char *) scan_arr_mem_19031_backing_1;\n    \n    int32_t x_17728;\n    int32_t x_17729;\n    int32_t x_17730;\n    int32_t x_17731;\n    \n    x_17728 = 0;\n    x_17729 = 0;\n    for (int32_t j_19033 = 0; j_19033 < squot32(aoa_len_17711 +\n                                                num_threads_19014 - 1,\n                                                num_threads_19014); j_19033++) {\n        int32_t chunk_offset_19034 = segscan_group_sizze_18060 * j_19033 +\n                group_tid_19026 * (segscan_group_sizze_18060 *\n                                   squot32(aoa_len_17711 + num_threads_19014 -\n                                           1, num_threads_19014));\n        int32_t flat_idx_19035 = chunk_offset_19034 + local_tid_19025;\n        int32_t gtid_18064 = flat_idx_19035;\n        \n        // threads in bounds read input; others get neutral element\n        {\n            if (slt32(gtid_18064, aoa_len_17711)) {\n                int32_t x_17737 = ((__global int32_t *) mem_18374)[gtid_18064];\n                \n                // write to-scan values to parameters\n                {\n                    x_17730 = x_17737;\n                    x_17731 = x_17737;\n                }\n                // write mapped values results to global memory\n                { }\n            } else {\n                x_17730 = 0;\n                x_17731 = 0;\n            }\n        }\n        // combine with carry and write to local memory\n        {\n            int32_t f_17732 = x_17728 | x_17730;\n            bool cond_17733 = x_17730 == 0;\n          ",
                   "  bool cond_17734 = !cond_17733;\n            int32_t res_17735;\n            \n            if (cond_17734) {\n                res_17735 = x_17731;\n            } else {\n                int32_t res_17736 = x_17729 + x_17731;\n                \n                res_17735 = res_17736;\n            }\n            ((__local int32_t *) scan_arr_mem_19029)[local_tid_19025] = f_17732;\n            ((__local int32_t *) scan_arr_mem_19031)[local_tid_19025] =\n                res_17735;\n        }\n        \n        int32_t x_19015;\n        int32_t x_19016;\n        int32_t x_19017;\n        int32_t x_19018;\n        int32_t x_19036;\n        int32_t x_19037;\n        int32_t x_19038;\n        int32_t x_19039;\n        int32_t skip_threads_19045;\n        \n        if (slt32(local_tid_19025, segscan_group_sizze_18060)) {\n            x_19017 = ((volatile __local\n                        int32_t *) scan_arr_mem_19029)[local_tid_19025];\n            x_19018 = ((volatile __local\n                        int32_t *) scan_arr_mem_19031)[local_tid_19025];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_19045 = 1;\n            while (slt32(skip_threads_19045, 32)) {\n                if (sle32(skip_threads_19045, local_tid_19025 -\n                          squot32(local_tid_19025, 32) * 32) &&\n                    slt32(local_tid_19025, segscan_group_sizze_18060)) {\n                    // read operands\n                    {\n                        x_19015 = ((volatile __local\n                                    int32_t *) scan_arr_mem_19029)[local_tid_19025 -\n                                                                   skip_threads_19045];\n                        x_19016 = ((volatile __local\n                                    int32_t *) scan_arr_mem_19031)[local_tid_19025 -\n                                                                   skip_threads_19045];\n                    }\n                    // perform operation\n                    {\n            ",
                   "            int32_t f_19019 = x_19015 | x_19017;\n                        bool cond_19020 = x_19017 == 0;\n                        bool cond_19021 = !cond_19020;\n                        int32_t res_19022;\n                        \n                        if (cond_19021) {\n                            res_19022 = x_19018;\n                        } else {\n                            int32_t res_19023 = x_19016 + x_19018;\n                            \n                            res_19022 = res_19023;\n                        }\n                        x_19017 = f_19019;\n                        x_19018 = res_19022;\n                    }\n                }\n                if (sle32(wave_sizze_19027, skip_threads_19045)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_19045, local_tid_19025 -\n                          squot32(local_tid_19025, 32) * 32) &&\n                    slt32(local_tid_19025, segscan_group_sizze_18060)) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_19029)[local_tid_19025] =\n                            x_19017;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_19031)[local_tid_19025] =\n                            x_19018;\n                    }\n                }\n                if (sle32(wave_sizze_19027, skip_threads_19045)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_19045 *= 2;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // last thread of block 'i' writes its result to offset 'i'\n        {\n            if ((local_tid_19025 - squot32(local_tid_19025, 32) * 32) == 31 &&\n                slt32(local_tid_19025, segscan_group_sizze_18060)) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_19029)[squot32(local_tid_19025, 32)] =\n                    x",
                   "_19017;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_19031)[squot32(local_tid_19025, 32)] =\n                    x_19018;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n        {\n            int32_t skip_threads_19046;\n            \n            if (squot32(local_tid_19025, 32) == 0 && slt32(local_tid_19025,\n                                                           segscan_group_sizze_18060)) {\n                x_19038 = ((volatile __local\n                            int32_t *) scan_arr_mem_19029)[local_tid_19025];\n                x_19039 = ((volatile __local\n                            int32_t *) scan_arr_mem_19031)[local_tid_19025];\n            }\n            // in-block scan (hopefully no barriers needed)\n            {\n                skip_threads_19046 = 1;\n                while (slt32(skip_threads_19046, 32)) {\n                    if (sle32(skip_threads_19046, local_tid_19025 -\n                              squot32(local_tid_19025, 32) * 32) &&\n                        (squot32(local_tid_19025, 32) == 0 &&\n                         slt32(local_tid_19025, segscan_group_sizze_18060))) {\n                        // read operands\n                        {\n                            x_19036 = ((volatile __local\n                                        int32_t *) scan_arr_mem_19029)[local_tid_19025 -\n                                                                       skip_threads_19046];\n                            x_19037 = ((volatile __local\n                                        int32_t *) scan_arr_mem_19031)[local_tid_19025 -\n                                                                       skip_threads_19046];\n                        }\n                        // perform operation\n                        {\n                            int32_t f_19040 = x_19036 | x_19038;\n                            bool cond_19041 = x_19038 == 0",
                   ";\n                            bool cond_19042 = !cond_19041;\n                            int32_t res_19043;\n                            \n                            if (cond_19042) {\n                                res_19043 = x_19039;\n                            } else {\n                                int32_t res_19044 = x_19037 + x_19039;\n                                \n                                res_19043 = res_19044;\n                            }\n                            x_19038 = f_19040;\n                            x_19039 = res_19043;\n                        }\n                    }\n                    if (sle32(wave_sizze_19027, skip_threads_19046)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    if (sle32(skip_threads_19046, local_tid_19025 -\n                              squot32(local_tid_19025, 32) * 32) &&\n                        (squot32(local_tid_19025, 32) == 0 &&\n                         slt32(local_tid_19025, segscan_group_sizze_18060))) {\n                        // write result\n                        {\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_19029)[local_tid_19025] =\n                                x_19038;\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_19031)[local_tid_19025] =\n                                x_19039;\n                        }\n                    }\n                    if (sle32(wave_sizze_19027, skip_threads_19046)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_19046 *= 2;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // carry-in for every block except the first\n        {\n            if (!(squot32(local_tid_19025, 32) == 0 || !slt32(local_tid_19025,\n                                                              segscan_group_sizze_18060))) {\n             ",
                   "   // read operands\n                {\n                    x_19015 = ((volatile __local\n                                int32_t *) scan_arr_mem_19029)[squot32(local_tid_19025,\n                                                                       32) - 1];\n                    x_19016 = ((volatile __local\n                                int32_t *) scan_arr_mem_19031)[squot32(local_tid_19025,\n                                                                       32) - 1];\n                }\n                // perform operation\n                {\n                    int32_t f_19019 = x_19015 | x_19017;\n                    bool cond_19020 = x_19017 == 0;\n                    bool cond_19021 = !cond_19020;\n                    int32_t res_19022;\n                    \n                    if (cond_19021) {\n                        res_19022 = x_19018;\n                    } else {\n                        int32_t res_19023 = x_19016 + x_19018;\n                        \n                        res_19022 = res_19023;\n                    }\n                    x_19017 = f_19019;\n                    x_19018 = res_19022;\n                }\n                // write final result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_19029)[local_tid_19025] = x_19017;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_19031)[local_tid_19025] = x_19018;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // restore correct values for first block\n        {\n            if (squot32(local_tid_19025, 32) == 0) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_19029)[local_tid_19025] = x_19017;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_19031)[local_tid_19025] = x_19018;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // threads in bounds write partial scan result\n        {\n            if (slt32(gtid",
                   "_18064, aoa_len_17711)) {\n                ((__global int32_t *) mem_18378)[gtid_18064] = ((__local\n                                                                 int32_t *) scan_arr_mem_19029)[local_tid_19025];\n                ((__global int32_t *) mem_18381)[gtid_18064] = ((__local\n                                                                 int32_t *) scan_arr_mem_19031)[local_tid_19025];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // first thread reads last element as carry-in for next iteration\n        {\n            if (local_tid_19025 == 0) {\n                x_17728 = ((__local\n                            int32_t *) scan_arr_mem_19029)[segscan_group_sizze_18060 -\n                                                           1];\n                x_17729 = ((__local\n                            int32_t *) scan_arr_mem_19031)[segscan_group_sizze_18060 -\n                                                           1];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n}\n__kernel void scan_stage1_18225(__local volatile\n                                int64_t *scan_arr_mem_19126_backing_aligned_0,\n                                int32_t convop_x_18393, __global\n                                unsigned char *mem_18399, __global\n                                unsigned char *mem_18403, __global\n                                unsigned char *mem_18406,\n                                int32_t num_threads_19117)\n{\n    const int32_t segscan_group_sizze_18220 = mainzisegscan_group_sizze_18219;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_19126_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_19126_backing_aligned_0;\n    int32_t global_tid_19121;\n    int32_t local_tid_19122;\n    int32_t group_sizze_19125;\n    int32_t wave_sizze_19124;\n    int32_t group_tid_19123;\n    \n    global_tid_1912",
                   "1 = get_global_id(0);\n    local_tid_19122 = get_local_id(0);\n    group_sizze_19125 = get_local_size(0);\n    wave_sizze_19124 = LOCKSTEP_WIDTH;\n    group_tid_19123 = get_group_id(0);\n    \n    int32_t phys_tid_18225 = global_tid_19121;\n    __local char *scan_arr_mem_19126;\n    \n    scan_arr_mem_19126 = (__local char *) scan_arr_mem_19126_backing_0;\n    \n    int32_t x_17790;\n    int32_t y_17791;\n    \n    x_17790 = 0;\n    for (int32_t j_19128 = 0; j_19128 < squot32(convop_x_18393 +\n                                                num_threads_19117 - 1,\n                                                num_threads_19117); j_19128++) {\n        int32_t chunk_offset_19129 = segscan_group_sizze_18220 * j_19128 +\n                group_tid_19123 * (segscan_group_sizze_18220 *\n                                   squot32(convop_x_18393 + num_threads_19117 -\n                                           1, num_threads_19117));\n        int32_t flat_idx_19130 = chunk_offset_19129 + local_tid_19122;\n        int32_t gtid_18224 = flat_idx_19130;\n        \n        // threads in bounds read input; others get neutral element\n        {\n            if (slt32(gtid_18224, convop_x_18393)) {\n                int32_t new_index_18259 = squot32(gtid_18224, 2);\n                int32_t binop_y_18261 = 2 * new_index_18259;\n                int32_t new_index_18262 = gtid_18224 - binop_y_18261;\n                int32_t x_17793 = ((__global\n                                    int32_t *) mem_18399)[new_index_18259 * 2 +\n                                                          new_index_18262];\n                bool res_17794 = x_17793 == 0;\n                bool res_17795 = !res_17794;\n                int32_t part_res_17796;\n                \n                if (res_17795) {\n                    part_res_17796 = 0;\n                } else {\n                    part_res_17796 = 1;\n                }\n                \n                int32_t part_res_17797;\n                \n                if (res_17795) {\n              ",
                   "      part_res_17797 = 1;\n                } else {\n                    part_res_17797 = 0;\n                }\n                // write to-scan values to parameters\n                {\n                    y_17791 = part_res_17797;\n                }\n                // write mapped values results to global memory\n                {\n                    ((__global int32_t *) mem_18406)[gtid_18224] =\n                        part_res_17796;\n                }\n            } else {\n                y_17791 = 0;\n            }\n        }\n        // combine with carry and write to local memory\n        {\n            int32_t zz_17792 = x_17790 + y_17791;\n            \n            ((__local int32_t *) scan_arr_mem_19126)[local_tid_19122] =\n                zz_17792;\n        }\n        \n        int32_t x_19118;\n        int32_t y_19119;\n        int32_t x_19131;\n        int32_t y_19132;\n        int32_t skip_threads_19134;\n        \n        if (slt32(local_tid_19122, segscan_group_sizze_18220)) {\n            y_19119 = ((volatile __local\n                        int32_t *) scan_arr_mem_19126)[local_tid_19122];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_19134 = 1;\n            while (slt32(skip_threads_19134, 32)) {\n                if (sle32(skip_threads_19134, local_tid_19122 -\n                          squot32(local_tid_19122, 32) * 32) &&\n                    slt32(local_tid_19122, segscan_group_sizze_18220)) {\n                    // read operands\n                    {\n                        x_19118 = ((volatile __local\n                                    int32_t *) scan_arr_mem_19126)[local_tid_19122 -\n                                                                   skip_threads_19134];\n                    }\n                    // perform operation\n                    {\n                        int32_t zz_19120 = x_19118 + y_19119;\n                        \n                        y_19119 = zz_19120;\n                    }\n                ",
                   "}\n                if (sle32(wave_sizze_19124, skip_threads_19134)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_19134, local_tid_19122 -\n                          squot32(local_tid_19122, 32) * 32) &&\n                    slt32(local_tid_19122, segscan_group_sizze_18220)) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_19126)[local_tid_19122] =\n                            y_19119;\n                    }\n                }\n                if (sle32(wave_sizze_19124, skip_threads_19134)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_19134 *= 2;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // last thread of block 'i' writes its result to offset 'i'\n        {\n            if ((local_tid_19122 - squot32(local_tid_19122, 32) * 32) == 31 &&\n                slt32(local_tid_19122, segscan_group_sizze_18220)) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_19126)[squot32(local_tid_19122, 32)] =\n                    y_19119;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n        {\n            int32_t skip_threads_19135;\n            \n            if (squot32(local_tid_19122, 32) == 0 && slt32(local_tid_19122,\n                                                           segscan_group_sizze_18220)) {\n                y_19132 = ((volatile __local\n                            int32_t *) scan_arr_mem_19126)[local_tid_19122];\n            }\n            // in-block scan (hopefully no barriers needed)\n            {\n                skip_threads_19135 = 1;\n                while (slt32(skip_threads_19135, 32)) {\n                    if (sle32(skip_threads_19135, local_tid_19122 -\n                              squot32(local_ti",
                   "d_19122, 32) * 32) &&\n                        (squot32(local_tid_19122, 32) == 0 &&\n                         slt32(local_tid_19122, segscan_group_sizze_18220))) {\n                        // read operands\n                        {\n                            x_19131 = ((volatile __local\n                                        int32_t *) scan_arr_mem_19126)[local_tid_19122 -\n                                                                       skip_threads_19135];\n                        }\n                        // perform operation\n                        {\n                            int32_t zz_19133 = x_19131 + y_19132;\n                            \n                            y_19132 = zz_19133;\n                        }\n                    }\n                    if (sle32(wave_sizze_19124, skip_threads_19135)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    if (sle32(skip_threads_19135, local_tid_19122 -\n                              squot32(local_tid_19122, 32) * 32) &&\n                        (squot32(local_tid_19122, 32) == 0 &&\n                         slt32(local_tid_19122, segscan_group_sizze_18220))) {\n                        // write result\n                        {\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_19126)[local_tid_19122] =\n                                y_19132;\n                        }\n                    }\n                    if (sle32(wave_sizze_19124, skip_threads_19135)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_19135 *= 2;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // carry-in for every block except the first\n        {\n            if (!(squot32(local_tid_19122, 32) == 0 || !slt32(local_tid_19122,\n                                                              segscan_group_sizze_18220))) {\n                // read operand",
                   "s\n                {\n                    x_19118 = ((volatile __local\n                                int32_t *) scan_arr_mem_19126)[squot32(local_tid_19122,\n                                                                       32) - 1];\n                }\n                // perform operation\n                {\n                    int32_t zz_19120 = x_19118 + y_19119;\n                    \n                    y_19119 = zz_19120;\n                }\n                // write final result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_19126)[local_tid_19122] = y_19119;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // restore correct values for first block\n        {\n            if (squot32(local_tid_19122, 32) == 0) {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_19126)[local_tid_19122] = y_19119;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // threads in bounds write partial scan result\n        {\n            if (slt32(gtid_18224, convop_x_18393)) {\n                ((__global int32_t *) mem_18403)[gtid_18224] = ((__local\n                                                                 int32_t *) scan_arr_mem_19126)[local_tid_19122];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // first thread reads last element as carry-in for next iteration\n        {\n            if (local_tid_19122 == 0) {\n                x_17790 = ((__local\n                            int32_t *) scan_arr_mem_19126)[segscan_group_sizze_18220 -\n                                                           1];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n}\n__kernel void scan_stage2_17841(__local volatile\n                                int64_t *scan_arr_mem_18556_backing_aligned_0,\n                                __local volatile\n                                int64_t *scan_arr_mem_18558_backing_aligned_1,\n      ",
                   "                          int32_t sizze_17490, int32_t num_groups_17838,\n                                __global unsigned char *mem_18307, __global\n                                unsigned char *mem_18310,\n                                int32_t num_threads_18512)\n{\n    const int32_t segscan_group_sizze_17836 = mainzisegscan_group_sizze_17835;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_18556_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_18556_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_18558_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_18558_backing_aligned_1;\n    int32_t global_tid_18551;\n    int32_t local_tid_18552;\n    int32_t group_sizze_18555;\n    int32_t wave_sizze_18554;\n    int32_t group_tid_18553;\n    \n    global_tid_18551 = get_global_id(0);\n    local_tid_18552 = get_local_id(0);\n    group_sizze_18555 = get_local_size(0);\n    wave_sizze_18554 = LOCKSTEP_WIDTH;\n    group_tid_18553 = get_group_id(0);\n    \n    int32_t phys_tid_17841 = global_tid_18551;\n    __local char *scan_arr_mem_18556;\n    \n    scan_arr_mem_18556 = (__local char *) scan_arr_mem_18556_backing_0;\n    \n    __local char *scan_arr_mem_18558;\n    \n    scan_arr_mem_18558 = (__local char *) scan_arr_mem_18558_backing_1;\n    \n    int32_t flat_idx_18560 = (local_tid_18552 + 1) *\n            (segscan_group_sizze_17836 * squot32(sizze_17490 +\n                                                 num_threads_18512 - 1,\n                                                 num_threads_18512)) - 1;\n    int32_t gtid_17840 = flat_idx_18560;\n    \n    // threads in bound read carries; others get neutral element\n    {\n        if (slt32(gtid_17840, sizze_17490)) {\n            ((__local int32_t *) scan_arr_mem_18556)[local_tid_18552] =\n                ((__global int32_t *) mem_18307)[gt",
                   "id_17840];\n            ((__local int32_t *) scan_arr_mem_18558)[local_tid_18552] =\n                ((__global int32_t *) mem_18310)[gtid_17840];\n        } else {\n            ((__local int32_t *) scan_arr_mem_18556)[local_tid_18552] = 0;\n            ((__local int32_t *) scan_arr_mem_18558)[local_tid_18552] = 0;\n        }\n    }\n    \n    int32_t x_18539;\n    int32_t x_18540;\n    int32_t x_18541;\n    int32_t x_18542;\n    int32_t x_18561;\n    int32_t x_18562;\n    int32_t x_18563;\n    int32_t x_18564;\n    int32_t skip_threads_18567;\n    \n    if (slt32(local_tid_18552, num_groups_17838)) {\n        x_18541 = ((volatile __local\n                    int32_t *) scan_arr_mem_18556)[local_tid_18552];\n        x_18542 = ((volatile __local\n                    int32_t *) scan_arr_mem_18558)[local_tid_18552];\n    }\n    // in-block scan (hopefully no barriers needed)\n    {\n        skip_threads_18567 = 1;\n        while (slt32(skip_threads_18567, 32)) {\n            if (sle32(skip_threads_18567, local_tid_18552 -\n                      squot32(local_tid_18552, 32) * 32) &&\n                slt32(local_tid_18552, num_groups_17838)) {\n                // read operands\n                {\n                    x_18539 = ((volatile __local\n                                int32_t *) scan_arr_mem_18556)[local_tid_18552 -\n                                                               skip_threads_18567];\n                    x_18540 = ((volatile __local\n                                int32_t *) scan_arr_mem_18558)[local_tid_18552 -\n                                                               skip_threads_18567];\n                }\n                // perform operation\n                {\n                    int32_t res_18543 = x_18539 + x_18541;\n                    int32_t res_18544 = x_18540 + x_18542;\n                    \n                    x_18541 = res_18543;\n                    x_18542 = res_18544;\n                }\n            }\n            if (sle32(wave_sizze_18554, skip_threads_18567)) {\n      ",
                   "          barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (sle32(skip_threads_18567, local_tid_18552 -\n                      squot32(local_tid_18552, 32) * 32) &&\n                slt32(local_tid_18552, num_groups_17838)) {\n                // write result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18556)[local_tid_18552] = x_18541;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18558)[local_tid_18552] = x_18542;\n                }\n            }\n            if (sle32(wave_sizze_18554, skip_threads_18567)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_threads_18567 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i' writes its result to offset 'i'\n    {\n        if ((local_tid_18552 - squot32(local_tid_18552, 32) * 32) == 31 &&\n            slt32(local_tid_18552, num_groups_17838)) {\n            ((volatile __local\n              int32_t *) scan_arr_mem_18556)[squot32(local_tid_18552, 32)] =\n                x_18541;\n            ((volatile __local\n              int32_t *) scan_arr_mem_18558)[squot32(local_tid_18552, 32)] =\n                x_18542;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n    {\n        int32_t skip_threads_18568;\n        \n        if (squot32(local_tid_18552, 32) == 0 && slt32(local_tid_18552,\n                                                       num_groups_17838)) {\n            x_18563 = ((volatile __local\n                        int32_t *) scan_arr_mem_18556)[local_tid_18552];\n            x_18564 = ((volatile __local\n                        int32_t *) scan_arr_mem_18558)[local_tid_18552];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_18568 = 1;\n            while (slt32(skip_threads_18568, 32)) {\n                if (sle32(skip_threads_18",
                   "568, local_tid_18552 -\n                          squot32(local_tid_18552, 32) * 32) &&\n                    (squot32(local_tid_18552, 32) == 0 && slt32(local_tid_18552,\n                                                                num_groups_17838))) {\n                    // read operands\n                    {\n                        x_18561 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18556)[local_tid_18552 -\n                                                                   skip_threads_18568];\n                        x_18562 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18558)[local_tid_18552 -\n                                                                   skip_threads_18568];\n                    }\n                    // perform operation\n                    {\n                        int32_t res_18565 = x_18561 + x_18563;\n                        int32_t res_18566 = x_18562 + x_18564;\n                        \n                        x_18563 = res_18565;\n                        x_18564 = res_18566;\n                    }\n                }\n                if (sle32(wave_sizze_18554, skip_threads_18568)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_18568, local_tid_18552 -\n                          squot32(local_tid_18552, 32) * 32) &&\n                    (squot32(local_tid_18552, 32) == 0 && slt32(local_tid_18552,\n                                                                num_groups_17838))) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18556)[local_tid_18552] =\n                            x_18563;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18558)[local_tid_18552] =\n                            x_18564;\n                    }\n                }\n                if (sle32(wav",
                   "e_sizze_18554, skip_threads_18568)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_18568 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_18552, 32) == 0 || !slt32(local_tid_18552,\n                                                          num_groups_17838))) {\n            // read operands\n            {\n                x_18539 = ((volatile __local\n                            int32_t *) scan_arr_mem_18556)[squot32(local_tid_18552,\n                                                                   32) - 1];\n                x_18540 = ((volatile __local\n                            int32_t *) scan_arr_mem_18558)[squot32(local_tid_18552,\n                                                                   32) - 1];\n            }\n            // perform operation\n            {\n                int32_t res_18543 = x_18539 + x_18541;\n                int32_t res_18544 = x_18540 + x_18542;\n                \n                x_18541 = res_18543;\n                x_18542 = res_18544;\n            }\n            // write final result\n            {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18556)[local_tid_18552] = x_18541;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18558)[local_tid_18552] = x_18542;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_18552, 32) == 0) {\n            ((volatile __local int32_t *) scan_arr_mem_18556)[local_tid_18552] =\n                x_18541;\n            ((volatile __local int32_t *) scan_arr_mem_18558)[local_tid_18552] =\n                x_18542;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // threads in bounds write scanned carries\n    {\n        if (slt32(gtid_17840, sizze_17490)) {\n            ((__global int32_t *) mem_18307)[gtid_",
                   "17840] = ((__local\n                                                             int32_t *) scan_arr_mem_18556)[local_tid_18552];\n            ((__global int32_t *) mem_18310)[gtid_17840] = ((__local\n                                                             int32_t *) scan_arr_mem_18558)[local_tid_18552];\n        }\n    }\n}\n__kernel void scan_stage2_17862(__local volatile\n                                int64_t *scan_arr_mem_18634_backing_aligned_0,\n                                __local volatile\n                                int64_t *scan_arr_mem_18636_backing_aligned_1,\n                                int32_t aoa_len_17514, int32_t num_groups_17859,\n                                __global unsigned char *mem_18317, __global\n                                unsigned char *mem_18320,\n                                int32_t num_threads_18582)\n{\n    const int32_t segscan_group_sizze_17857 = mainzisegscan_group_sizze_17856;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_18634_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_18634_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_18636_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_18636_backing_aligned_1;\n    int32_t global_tid_18629;\n    int32_t local_tid_18630;\n    int32_t group_sizze_18633;\n    int32_t wave_sizze_18632;\n    int32_t group_tid_18631;\n    \n    global_tid_18629 = get_global_id(0);\n    local_tid_18630 = get_local_id(0);\n    group_sizze_18633 = get_local_size(0);\n    wave_sizze_18632 = LOCKSTEP_WIDTH;\n    group_tid_18631 = get_group_id(0);\n    \n    int32_t phys_tid_17862 = global_tid_18629;\n    __local char *scan_arr_mem_18634;\n    \n    scan_arr_mem_18634 = (__local char *) scan_arr_mem_18634_backing_0;\n    \n    __local char *scan_arr_mem_18636;\n    \n    scan_arr_mem_18636 = (__local c",
                   "har *) scan_arr_mem_18636_backing_1;\n    \n    int32_t flat_idx_18638 = (local_tid_18630 + 1) *\n            (segscan_group_sizze_17857 * squot32(aoa_len_17514 +\n                                                 num_threads_18582 - 1,\n                                                 num_threads_18582)) - 1;\n    int32_t gtid_17861 = flat_idx_18638;\n    \n    // threads in bound read carries; others get neutral element\n    {\n        if (slt32(gtid_17861, aoa_len_17514)) {\n            ((__local int32_t *) scan_arr_mem_18634)[local_tid_18630] =\n                ((__global int32_t *) mem_18317)[gtid_17861];\n            ((__local int32_t *) scan_arr_mem_18636)[local_tid_18630] =\n                ((__global int32_t *) mem_18320)[gtid_17861];\n        } else {\n            ((__local int32_t *) scan_arr_mem_18634)[local_tid_18630] = 0;\n            ((__local int32_t *) scan_arr_mem_18636)[local_tid_18630] = 0;\n        }\n    }\n    \n    int32_t x_18613;\n    int32_t x_18614;\n    int32_t x_18615;\n    int32_t x_18616;\n    int32_t x_18639;\n    int32_t x_18640;\n    int32_t x_18641;\n    int32_t x_18642;\n    int32_t skip_threads_18647;\n    \n    if (slt32(local_tid_18630, num_groups_17859)) {\n        x_18615 = ((volatile __local\n                    int32_t *) scan_arr_mem_18634)[local_tid_18630];\n        x_18616 = ((volatile __local\n                    int32_t *) scan_arr_mem_18636)[local_tid_18630];\n    }\n    // in-block scan (hopefully no barriers needed)\n    {\n        skip_threads_18647 = 1;\n        while (slt32(skip_threads_18647, 32)) {\n            if (sle32(skip_threads_18647, local_tid_18630 -\n                      squot32(local_tid_18630, 32) * 32) &&\n                slt32(local_tid_18630, num_groups_17859)) {\n                // read operands\n                {\n                    x_18613 = ((volatile __local\n                                int32_t *) scan_arr_mem_18634)[local_tid_18630 -\n                                                               skip_threads_18647];\n               ",
                   "     x_18614 = ((volatile __local\n                                int32_t *) scan_arr_mem_18636)[local_tid_18630 -\n                                                               skip_threads_18647];\n                }\n                // perform operation\n                {\n                    int32_t f_18617 = x_18613 | x_18615;\n                    bool cond_18618 = slt32(0, x_18615);\n                    int32_t res_18619;\n                    \n                    if (cond_18618) {\n                        res_18619 = x_18616;\n                    } else {\n                        int32_t res_18620 = x_18614 + x_18616;\n                        \n                        res_18619 = res_18620;\n                    }\n                    x_18615 = f_18617;\n                    x_18616 = res_18619;\n                }\n            }\n            if (sle32(wave_sizze_18632, skip_threads_18647)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (sle32(skip_threads_18647, local_tid_18630 -\n                      squot32(local_tid_18630, 32) * 32) &&\n                slt32(local_tid_18630, num_groups_17859)) {\n                // write result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18634)[local_tid_18630] = x_18615;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18636)[local_tid_18630] = x_18616;\n                }\n            }\n            if (sle32(wave_sizze_18632, skip_threads_18647)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_threads_18647 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i' writes its result to offset 'i'\n    {\n        if ((local_tid_18630 - squot32(local_tid_18630, 32) * 32) == 31 &&\n            slt32(local_tid_18630, num_groups_17859)) {\n            ((volatile __local\n              int32_t *) scan_arr_mem_18634)[squot32(local_tid_18630, 32)] =\n                x_18615;\n    ",
                   "        ((volatile __local\n              int32_t *) scan_arr_mem_18636)[squot32(local_tid_18630, 32)] =\n                x_18616;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n    {\n        int32_t skip_threads_18648;\n        \n        if (squot32(local_tid_18630, 32) == 0 && slt32(local_tid_18630,\n                                                       num_groups_17859)) {\n            x_18641 = ((volatile __local\n                        int32_t *) scan_arr_mem_18634)[local_tid_18630];\n            x_18642 = ((volatile __local\n                        int32_t *) scan_arr_mem_18636)[local_tid_18630];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_18648 = 1;\n            while (slt32(skip_threads_18648, 32)) {\n                if (sle32(skip_threads_18648, local_tid_18630 -\n                          squot32(local_tid_18630, 32) * 32) &&\n                    (squot32(local_tid_18630, 32) == 0 && slt32(local_tid_18630,\n                                                                num_groups_17859))) {\n                    // read operands\n                    {\n                        x_18639 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18634)[local_tid_18630 -\n                                                                   skip_threads_18648];\n                        x_18640 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18636)[local_tid_18630 -\n                                                                   skip_threads_18648];\n                    }\n                    // perform operation\n                    {\n                        int32_t f_18643 = x_18639 | x_18641;\n                        bool cond_18644 = slt32(0, x_18641);\n                        int32_t res_18645;\n                        \n                        if (cond_18644) {\n                      ",
                   "      res_18645 = x_18642;\n                        } else {\n                            int32_t res_18646 = x_18640 + x_18642;\n                            \n                            res_18645 = res_18646;\n                        }\n                        x_18641 = f_18643;\n                        x_18642 = res_18645;\n                    }\n                }\n                if (sle32(wave_sizze_18632, skip_threads_18648)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_18648, local_tid_18630 -\n                          squot32(local_tid_18630, 32) * 32) &&\n                    (squot32(local_tid_18630, 32) == 0 && slt32(local_tid_18630,\n                                                                num_groups_17859))) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18634)[local_tid_18630] =\n                            x_18641;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18636)[local_tid_18630] =\n                            x_18642;\n                    }\n                }\n                if (sle32(wave_sizze_18632, skip_threads_18648)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_18648 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_18630, 32) == 0 || !slt32(local_tid_18630,\n                                                          num_groups_17859))) {\n            // read operands\n            {\n                x_18613 = ((volatile __local\n                            int32_t *) scan_arr_mem_18634)[squot32(local_tid_18630,\n                                                                   32) - 1];\n                x_18614 = ((volatile __local\n                            int32_t *) scan_arr_mem_186",
                   "36)[squot32(local_tid_18630,\n                                                                   32) - 1];\n            }\n            // perform operation\n            {\n                int32_t f_18617 = x_18613 | x_18615;\n                bool cond_18618 = slt32(0, x_18615);\n                int32_t res_18619;\n                \n                if (cond_18618) {\n                    res_18619 = x_18616;\n                } else {\n                    int32_t res_18620 = x_18614 + x_18616;\n                    \n                    res_18619 = res_18620;\n                }\n                x_18615 = f_18617;\n                x_18616 = res_18619;\n            }\n            // write final result\n            {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18634)[local_tid_18630] = x_18615;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18636)[local_tid_18630] = x_18616;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_18630, 32) == 0) {\n            ((volatile __local int32_t *) scan_arr_mem_18634)[local_tid_18630] =\n                x_18615;\n            ((volatile __local int32_t *) scan_arr_mem_18636)[local_tid_18630] =\n                x_18616;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // threads in bounds write scanned carries\n    {\n        if (slt32(gtid_17861, aoa_len_17514)) {\n            ((__global int32_t *) mem_18317)[gtid_17861] = ((__local\n                                                             int32_t *) scan_arr_mem_18634)[local_tid_18630];\n            ((__global int32_t *) mem_18320)[gtid_17861] = ((__local\n                                                             int32_t *) scan_arr_mem_18636)[local_tid_18630];\n        }\n    }\n}\n__kernel void scan_stage2_17871(__local volatile\n                                int64_t *scan_arr_mem_18687_backing_aligned_0,\n                                int32_t sizz",
                   "e_17490, int32_t num_groups_17868,\n                                __global unsigned char *mem_18324,\n                                int32_t num_threads_18657)\n{\n    const int32_t segscan_group_sizze_17866 = mainzisegscan_group_sizze_17865;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_18687_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_18687_backing_aligned_0;\n    int32_t global_tid_18682;\n    int32_t local_tid_18683;\n    int32_t group_sizze_18686;\n    int32_t wave_sizze_18685;\n    int32_t group_tid_18684;\n    \n    global_tid_18682 = get_global_id(0);\n    local_tid_18683 = get_local_id(0);\n    group_sizze_18686 = get_local_size(0);\n    wave_sizze_18685 = LOCKSTEP_WIDTH;\n    group_tid_18684 = get_group_id(0);\n    \n    int32_t phys_tid_17871 = global_tid_18682;\n    __local char *scan_arr_mem_18687;\n    \n    scan_arr_mem_18687 = (__local char *) scan_arr_mem_18687_backing_0;\n    \n    int32_t flat_idx_18689 = (local_tid_18683 + 1) *\n            (segscan_group_sizze_17866 * squot32(sizze_17490 +\n                                                 num_threads_18657 - 1,\n                                                 num_threads_18657)) - 1;\n    int32_t gtid_17870 = flat_idx_18689;\n    \n    // threads in bound read carries; others get neutral element\n    {\n        if (slt32(gtid_17870, sizze_17490)) {\n            ((__local int32_t *) scan_arr_mem_18687)[local_tid_18683] =\n                ((__global int32_t *) mem_18324)[gtid_17870];\n        } else {\n            ((__local int32_t *) scan_arr_mem_18687)[local_tid_18683] = 0;\n        }\n    }\n    \n    int32_t x_18676;\n    int32_t x_18677;\n    int32_t x_18690;\n    int32_t x_18691;\n    int32_t skip_threads_18693;\n    \n    if (slt32(local_tid_18683, num_groups_17868)) {\n        x_18677 = ((volatile __local\n                    int32_t *) scan_arr_mem_18687)[local_tid_18683];\n    }\n",
                   "    // in-block scan (hopefully no barriers needed)\n    {\n        skip_threads_18693 = 1;\n        while (slt32(skip_threads_18693, 32)) {\n            if (sle32(skip_threads_18693, local_tid_18683 -\n                      squot32(local_tid_18683, 32) * 32) &&\n                slt32(local_tid_18683, num_groups_17868)) {\n                // read operands\n                {\n                    x_18676 = ((volatile __local\n                                int32_t *) scan_arr_mem_18687)[local_tid_18683 -\n                                                               skip_threads_18693];\n                }\n                // perform operation\n                {\n                    int32_t res_18678 = x_18676 + x_18677;\n                    \n                    x_18677 = res_18678;\n                }\n            }\n            if (sle32(wave_sizze_18685, skip_threads_18693)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (sle32(skip_threads_18693, local_tid_18683 -\n                      squot32(local_tid_18683, 32) * 32) &&\n                slt32(local_tid_18683, num_groups_17868)) {\n                // write result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18687)[local_tid_18683] = x_18677;\n                }\n            }\n            if (sle32(wave_sizze_18685, skip_threads_18693)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_threads_18693 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i' writes its result to offset 'i'\n    {\n        if ((local_tid_18683 - squot32(local_tid_18683, 32) * 32) == 31 &&\n            slt32(local_tid_18683, num_groups_17868)) {\n            ((volatile __local\n              int32_t *) scan_arr_mem_18687)[squot32(local_tid_18683, 32)] =\n                x_18677;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n    {\n ",
                   "       int32_t skip_threads_18694;\n        \n        if (squot32(local_tid_18683, 32) == 0 && slt32(local_tid_18683,\n                                                       num_groups_17868)) {\n            x_18691 = ((volatile __local\n                        int32_t *) scan_arr_mem_18687)[local_tid_18683];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_18694 = 1;\n            while (slt32(skip_threads_18694, 32)) {\n                if (sle32(skip_threads_18694, local_tid_18683 -\n                          squot32(local_tid_18683, 32) * 32) &&\n                    (squot32(local_tid_18683, 32) == 0 && slt32(local_tid_18683,\n                                                                num_groups_17868))) {\n                    // read operands\n                    {\n                        x_18690 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18687)[local_tid_18683 -\n                                                                   skip_threads_18694];\n                    }\n                    // perform operation\n                    {\n                        int32_t res_18692 = x_18690 + x_18691;\n                        \n                        x_18691 = res_18692;\n                    }\n                }\n                if (sle32(wave_sizze_18685, skip_threads_18694)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_18694, local_tid_18683 -\n                          squot32(local_tid_18683, 32) * 32) &&\n                    (squot32(local_tid_18683, 32) == 0 && slt32(local_tid_18683,\n                                                                num_groups_17868))) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18687)[local_tid_18683] =\n                            x_18691;\n                    }\n                }\n        ",
                   "        if (sle32(wave_sizze_18685, skip_threads_18694)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_18694 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_18683, 32) == 0 || !slt32(local_tid_18683,\n                                                          num_groups_17868))) {\n            // read operands\n            {\n                x_18676 = ((volatile __local\n                            int32_t *) scan_arr_mem_18687)[squot32(local_tid_18683,\n                                                                   32) - 1];\n            }\n            // perform operation\n            {\n                int32_t res_18678 = x_18676 + x_18677;\n                \n                x_18677 = res_18678;\n            }\n            // write final result\n            {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18687)[local_tid_18683] = x_18677;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_18683, 32) == 0) {\n            ((volatile __local int32_t *) scan_arr_mem_18687)[local_tid_18683] =\n                x_18677;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // threads in bounds write scanned carries\n    {\n        if (slt32(gtid_17870, sizze_17490)) {\n            ((__global int32_t *) mem_18324)[gtid_17870] = ((__local\n                                                             int32_t *) scan_arr_mem_18687)[local_tid_18683];\n        }\n    }\n}\n__kernel void scan_stage2_17960(__local volatile\n                                int64_t *scan_arr_mem_18837_backing_aligned_0,\n                                __local volatile\n                                int64_t *scan_arr_mem_18839_backing_aligned_1,\n                                __local volatile\n                                int64_t *sc",
                   "an_arr_mem_18841_backing_aligned_2,\n                                __local volatile\n                                int64_t *scan_arr_mem_18843_backing_aligned_3,\n                                __local volatile\n                                int64_t *scan_arr_mem_18845_backing_aligned_4,\n                                __local volatile\n                                int64_t *scan_arr_mem_18847_backing_aligned_5,\n                                int32_t aoa_len_17582, int32_t num_groups_17957,\n                                __global unsigned char *mem_18339, __global\n                                unsigned char *mem_18342, __global\n                                unsigned char *mem_18345, __global\n                                unsigned char *mem_18348, __global\n                                unsigned char *mem_18351, __global\n                                unsigned char *mem_18354,\n                                int32_t num_threads_18713)\n{\n    const int32_t segscan_group_sizze_17955 = mainzisegscan_group_sizze_17954;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_18837_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_18837_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_18839_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_18839_backing_aligned_1;\n    __local volatile char *restrict scan_arr_mem_18841_backing_2 =\n                          (__local volatile\n                           char *) scan_arr_mem_18841_backing_aligned_2;\n    __local volatile char *restrict scan_arr_mem_18843_backing_3 =\n                          (__local volatile\n                           char *) scan_arr_mem_18843_backing_aligned_3;\n    __local volatile char *restrict scan_arr_mem_18845_backing_4 =\n                          (__local volatile\n                    ",
                   "       char *) scan_arr_mem_18845_backing_aligned_4;\n    __local volatile char *restrict scan_arr_mem_18847_backing_5 =\n                          (__local volatile\n                           char *) scan_arr_mem_18847_backing_aligned_5;\n    int32_t global_tid_18832;\n    int32_t local_tid_18833;\n    int32_t group_sizze_18836;\n    int32_t wave_sizze_18835;\n    int32_t group_tid_18834;\n    \n    global_tid_18832 = get_global_id(0);\n    local_tid_18833 = get_local_id(0);\n    group_sizze_18836 = get_local_size(0);\n    wave_sizze_18835 = LOCKSTEP_WIDTH;\n    group_tid_18834 = get_group_id(0);\n    \n    int32_t phys_tid_17960 = global_tid_18832;\n    __local char *scan_arr_mem_18837;\n    \n    scan_arr_mem_18837 = (__local char *) scan_arr_mem_18837_backing_0;\n    \n    __local char *scan_arr_mem_18839;\n    \n    scan_arr_mem_18839 = (__local char *) scan_arr_mem_18839_backing_1;\n    \n    __local char *scan_arr_mem_18841;\n    \n    scan_arr_mem_18841 = (__local char *) scan_arr_mem_18841_backing_2;\n    \n    __local char *scan_arr_mem_18843;\n    \n    scan_arr_mem_18843 = (__local char *) scan_arr_mem_18843_backing_3;\n    \n    __local char *scan_arr_mem_18845;\n    \n    scan_arr_mem_18845 = (__local char *) scan_arr_mem_18845_backing_4;\n    \n    __local char *scan_arr_mem_18847;\n    \n    scan_arr_mem_18847 = (__local char *) scan_arr_mem_18847_backing_5;\n    \n    int32_t flat_idx_18849 = (local_tid_18833 + 1) *\n            (segscan_group_sizze_17955 * squot32(aoa_len_17582 +\n                                                 num_threads_18713 - 1,\n                                                 num_threads_18713)) - 1;\n    int32_t gtid_17959 = flat_idx_18849;\n    \n    // threads in bound read carries; others get neutral element\n    {\n        if (slt32(gtid_17959, aoa_len_17582)) {\n            ((__local int32_t *) scan_arr_mem_18837)[local_tid_18833] =\n                ((__global int32_t *) mem_18339)[gtid_17959];\n            ((__local int32_t *) scan_arr_mem_18839)[local_tid_18833] =\n ",
                   "               ((__global int32_t *) mem_18342)[gtid_17959];\n            ((__local int32_t *) scan_arr_mem_18841)[local_tid_18833] =\n                ((__global int32_t *) mem_18345)[gtid_17959];\n            ((__local int32_t *) scan_arr_mem_18843)[local_tid_18833] =\n                ((__global int32_t *) mem_18348)[gtid_17959];\n            ((__local int32_t *) scan_arr_mem_18845)[local_tid_18833] =\n                ((__global int32_t *) mem_18351)[gtid_17959];\n            ((__local int32_t *) scan_arr_mem_18847)[local_tid_18833] =\n                ((__global int32_t *) mem_18354)[gtid_17959];\n        } else {\n            ((__local int32_t *) scan_arr_mem_18837)[local_tid_18833] = 0;\n            ((__local int32_t *) scan_arr_mem_18839)[local_tid_18833] = 0;\n            ((__local int32_t *) scan_arr_mem_18841)[local_tid_18833] = 0;\n            ((__local int32_t *) scan_arr_mem_18843)[local_tid_18833] = 0;\n            ((__local int32_t *) scan_arr_mem_18845)[local_tid_18833] = 0;\n            ((__local int32_t *) scan_arr_mem_18847)[local_tid_18833] = 0;\n        }\n    }\n    \n    int32_t x_18784;\n    int32_t x_18785;\n    int32_t x_18786;\n    int32_t x_18787;\n    int32_t x_18788;\n    int32_t x_18789;\n    int32_t x_18790;\n    int32_t x_18791;\n    int32_t x_18792;\n    int32_t x_18793;\n    int32_t x_18794;\n    int32_t x_18795;\n    int32_t x_18850;\n    int32_t x_18851;\n    int32_t x_18852;\n    int32_t x_18853;\n    int32_t x_18854;\n    int32_t x_18855;\n    int32_t x_18856;\n    int32_t x_18857;\n    int32_t x_18858;\n    int32_t x_18859;\n    int32_t x_18860;\n    int32_t x_18861;\n    int32_t skip_threads_18874;\n    \n    if (slt32(local_tid_18833, num_groups_17957)) {\n        x_18790 = ((volatile __local\n                    int32_t *) scan_arr_mem_18837)[local_tid_18833];\n        x_18791 = ((volatile __local\n                    int32_t *) scan_arr_mem_18839)[local_tid_18833];\n        x_18792 = ((volatile __local\n                    int32_t *) scan_arr_mem_18841)[local_tid_18833];\n    ",
                   "    x_18793 = ((volatile __local\n                    int32_t *) scan_arr_mem_18843)[local_tid_18833];\n        x_18794 = ((volatile __local\n                    int32_t *) scan_arr_mem_18845)[local_tid_18833];\n        x_18795 = ((volatile __local\n                    int32_t *) scan_arr_mem_18847)[local_tid_18833];\n    }\n    // in-block scan (hopefully no barriers needed)\n    {\n        skip_threads_18874 = 1;\n        while (slt32(skip_threads_18874, 32)) {\n            if (sle32(skip_threads_18874, local_tid_18833 -\n                      squot32(local_tid_18833, 32) * 32) &&\n                slt32(local_tid_18833, num_groups_17957)) {\n                // read operands\n                {\n                    x_18784 = ((volatile __local\n                                int32_t *) scan_arr_mem_18837)[local_tid_18833 -\n                                                               skip_threads_18874];\n                    x_18785 = ((volatile __local\n                                int32_t *) scan_arr_mem_18839)[local_tid_18833 -\n                                                               skip_threads_18874];\n                    x_18786 = ((volatile __local\n                                int32_t *) scan_arr_mem_18841)[local_tid_18833 -\n                                                               skip_threads_18874];\n                    x_18787 = ((volatile __local\n                                int32_t *) scan_arr_mem_18843)[local_tid_18833 -\n                                                               skip_threads_18874];\n                    x_18788 = ((volatile __local\n                                int32_t *) scan_arr_mem_18845)[local_tid_18833 -\n                                                               skip_threads_18874];\n                    x_18789 = ((volatile __local\n                                int32_t *) scan_arr_mem_18847)[local_tid_18833 -\n                                                               skip_threads_18874];\n                }\n         ",
                   "       // perform operation\n                {\n                    int32_t f_18796 = x_18784 | x_18790;\n                    bool cond_18797 = slt32(0, x_18790);\n                    int32_t res_18798;\n                    \n                    if (cond_18797) {\n                        res_18798 = x_18791;\n                    } else {\n                        int32_t res_18799 = x_18785 + x_18791;\n                        \n                        res_18798 = res_18799;\n                    }\n                    \n                    int32_t f_18800 = x_18786 | x_18792;\n                    bool cond_18801 = slt32(0, x_18792);\n                    int32_t res_18802;\n                    \n                    if (cond_18801) {\n                        res_18802 = x_18793;\n                    } else {\n                        int32_t res_18803 = x_18787 + x_18793;\n                        \n                        res_18802 = res_18803;\n                    }\n                    \n                    int32_t f_18804 = x_18788 | x_18794;\n                    bool cond_18805 = slt32(0, x_18794);\n                    int32_t res_18806;\n                    \n                    if (cond_18805) {\n                        res_18806 = x_18795;\n                    } else {\n                        int32_t res_18807 = x_18789 + x_18795;\n                        \n                        res_18806 = res_18807;\n                    }\n                    x_18790 = f_18796;\n                    x_18791 = res_18798;\n                    x_18792 = f_18800;\n                    x_18793 = res_18802;\n                    x_18794 = f_18804;\n                    x_18795 = res_18806;\n                }\n            }\n            if (sle32(wave_sizze_18835, skip_threads_18874)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (sle32(skip_threads_18874, local_tid_18833 -\n                      squot32(local_tid_18833, 32) * 32) &&\n                slt32(local_tid_18833, num_groups_17957)) {\n        ",
                   "        // write result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18837)[local_tid_18833] = x_18790;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18839)[local_tid_18833] = x_18791;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18841)[local_tid_18833] = x_18792;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18843)[local_tid_18833] = x_18793;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18845)[local_tid_18833] = x_18794;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18847)[local_tid_18833] = x_18795;\n                }\n            }\n            if (sle32(wave_sizze_18835, skip_threads_18874)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_threads_18874 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i' writes its result to offset 'i'\n    {\n        if ((local_tid_18833 - squot32(local_tid_18833, 32) * 32) == 31 &&\n            slt32(local_tid_18833, num_groups_17957)) {\n            ((volatile __local\n              int32_t *) scan_arr_mem_18837)[squot32(local_tid_18833, 32)] =\n                x_18790;\n            ((volatile __local\n              int32_t *) scan_arr_mem_18839)[squot32(local_tid_18833, 32)] =\n                x_18791;\n            ((volatile __local\n              int32_t *) scan_arr_mem_18841)[squot32(local_tid_18833, 32)] =\n                x_18792;\n            ((volatile __local\n              int32_t *) scan_arr_mem_18843)[squot32(local_tid_18833, 32)] =\n                x_18793;\n            ((volatile __local\n              int32_t *) scan_arr_mem_18845)[squot32(local_tid_18833, 32)] =\n                x_18794;\n            ((volatile __local\n              int32_t *) scan_arr_mem_18847)[squot32(local_tid_18833, 32)] =\n                x_1879",
                   "5;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n    {\n        int32_t skip_threads_18875;\n        \n        if (squot32(local_tid_18833, 32) == 0 && slt32(local_tid_18833,\n                                                       num_groups_17957)) {\n            x_18856 = ((volatile __local\n                        int32_t *) scan_arr_mem_18837)[local_tid_18833];\n            x_18857 = ((volatile __local\n                        int32_t *) scan_arr_mem_18839)[local_tid_18833];\n            x_18858 = ((volatile __local\n                        int32_t *) scan_arr_mem_18841)[local_tid_18833];\n            x_18859 = ((volatile __local\n                        int32_t *) scan_arr_mem_18843)[local_tid_18833];\n            x_18860 = ((volatile __local\n                        int32_t *) scan_arr_mem_18845)[local_tid_18833];\n            x_18861 = ((volatile __local\n                        int32_t *) scan_arr_mem_18847)[local_tid_18833];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_18875 = 1;\n            while (slt32(skip_threads_18875, 32)) {\n                if (sle32(skip_threads_18875, local_tid_18833 -\n                          squot32(local_tid_18833, 32) * 32) &&\n                    (squot32(local_tid_18833, 32) == 0 && slt32(local_tid_18833,\n                                                                num_groups_17957))) {\n                    // read operands\n                    {\n                        x_18850 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18837)[local_tid_18833 -\n                                                                   skip_threads_18875];\n                        x_18851 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18839)[local_tid_18833 -\n                                                                   skip_threads_18875];\n         ",
                   "               x_18852 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18841)[local_tid_18833 -\n                                                                   skip_threads_18875];\n                        x_18853 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18843)[local_tid_18833 -\n                                                                   skip_threads_18875];\n                        x_18854 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18845)[local_tid_18833 -\n                                                                   skip_threads_18875];\n                        x_18855 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18847)[local_tid_18833 -\n                                                                   skip_threads_18875];\n                    }\n                    // perform operation\n                    {\n                        int32_t f_18862 = x_18850 | x_18856;\n                        bool cond_18863 = slt32(0, x_18856);\n                        int32_t res_18864;\n                        \n                        if (cond_18863) {\n                            res_18864 = x_18857;\n                        } else {\n                            int32_t res_18865 = x_18851 + x_18857;\n                            \n                            res_18864 = res_18865;\n                        }\n                        \n                        int32_t f_18866 = x_18852 | x_18858;\n                        bool cond_18867 = slt32(0, x_18858);\n                        int32_t res_18868;\n                        \n                        if (cond_18867) {\n                            res_18868 = x_18859;\n                        } else {\n                            int32_t res_18869 = x_18853 + x_18859;\n                            \n                            res_18868 = res_18869;\n                        }\n                     ",
                   "   \n                        int32_t f_18870 = x_18854 | x_18860;\n                        bool cond_18871 = slt32(0, x_18860);\n                        int32_t res_18872;\n                        \n                        if (cond_18871) {\n                            res_18872 = x_18861;\n                        } else {\n                            int32_t res_18873 = x_18855 + x_18861;\n                            \n                            res_18872 = res_18873;\n                        }\n                        x_18856 = f_18862;\n                        x_18857 = res_18864;\n                        x_18858 = f_18866;\n                        x_18859 = res_18868;\n                        x_18860 = f_18870;\n                        x_18861 = res_18872;\n                    }\n                }\n                if (sle32(wave_sizze_18835, skip_threads_18875)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_18875, local_tid_18833 -\n                          squot32(local_tid_18833, 32) * 32) &&\n                    (squot32(local_tid_18833, 32) == 0 && slt32(local_tid_18833,\n                                                                num_groups_17957))) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18837)[local_tid_18833] =\n                            x_18856;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18839)[local_tid_18833] =\n                            x_18857;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18841)[local_tid_18833] =\n                            x_18858;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18843)[local_tid_18833] =\n                            x_18859;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_",
                   "18845)[local_tid_18833] =\n                            x_18860;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18847)[local_tid_18833] =\n                            x_18861;\n                    }\n                }\n                if (sle32(wave_sizze_18835, skip_threads_18875)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_18875 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_18833, 32) == 0 || !slt32(local_tid_18833,\n                                                          num_groups_17957))) {\n            // read operands\n            {\n                x_18784 = ((volatile __local\n                            int32_t *) scan_arr_mem_18837)[squot32(local_tid_18833,\n                                                                   32) - 1];\n                x_18785 = ((volatile __local\n                            int32_t *) scan_arr_mem_18839)[squot32(local_tid_18833,\n                                                                   32) - 1];\n                x_18786 = ((volatile __local\n                            int32_t *) scan_arr_mem_18841)[squot32(local_tid_18833,\n                                                                   32) - 1];\n                x_18787 = ((volatile __local\n                            int32_t *) scan_arr_mem_18843)[squot32(local_tid_18833,\n                                                                   32) - 1];\n                x_18788 = ((volatile __local\n                            int32_t *) scan_arr_mem_18845)[squot32(local_tid_18833,\n                                                                   32) - 1];\n                x_18789 = ((volatile __local\n                            int32_t *) scan_arr_mem_18847)[squot32(local_tid_18833,\n                                                                   32) - 1];\n            ",
                   "}\n            // perform operation\n            {\n                int32_t f_18796 = x_18784 | x_18790;\n                bool cond_18797 = slt32(0, x_18790);\n                int32_t res_18798;\n                \n                if (cond_18797) {\n                    res_18798 = x_18791;\n                } else {\n                    int32_t res_18799 = x_18785 + x_18791;\n                    \n                    res_18798 = res_18799;\n                }\n                \n                int32_t f_18800 = x_18786 | x_18792;\n                bool cond_18801 = slt32(0, x_18792);\n                int32_t res_18802;\n                \n                if (cond_18801) {\n                    res_18802 = x_18793;\n                } else {\n                    int32_t res_18803 = x_18787 + x_18793;\n                    \n                    res_18802 = res_18803;\n                }\n                \n                int32_t f_18804 = x_18788 | x_18794;\n                bool cond_18805 = slt32(0, x_18794);\n                int32_t res_18806;\n                \n                if (cond_18805) {\n                    res_18806 = x_18795;\n                } else {\n                    int32_t res_18807 = x_18789 + x_18795;\n                    \n                    res_18806 = res_18807;\n                }\n                x_18790 = f_18796;\n                x_18791 = res_18798;\n                x_18792 = f_18800;\n                x_18793 = res_18802;\n                x_18794 = f_18804;\n                x_18795 = res_18806;\n            }\n            // write final result\n            {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18837)[local_tid_18833] = x_18790;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18839)[local_tid_18833] = x_18791;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18841)[local_tid_18833] = x_18792;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18843)[local_tid_18833] = x_1",
                   "8793;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18845)[local_tid_18833] = x_18794;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18847)[local_tid_18833] = x_18795;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_18833, 32) == 0) {\n            ((volatile __local int32_t *) scan_arr_mem_18837)[local_tid_18833] =\n                x_18790;\n            ((volatile __local int32_t *) scan_arr_mem_18839)[local_tid_18833] =\n                x_18791;\n            ((volatile __local int32_t *) scan_arr_mem_18841)[local_tid_18833] =\n                x_18792;\n            ((volatile __local int32_t *) scan_arr_mem_18843)[local_tid_18833] =\n                x_18793;\n            ((volatile __local int32_t *) scan_arr_mem_18845)[local_tid_18833] =\n                x_18794;\n            ((volatile __local int32_t *) scan_arr_mem_18847)[local_tid_18833] =\n                x_18795;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // threads in bounds write scanned carries\n    {\n        if (slt32(gtid_17959, aoa_len_17582)) {\n            ((__global int32_t *) mem_18339)[gtid_17959] = ((__local\n                                                             int32_t *) scan_arr_mem_18837)[local_tid_18833];\n            ((__global int32_t *) mem_18342)[gtid_17959] = ((__local\n                                                             int32_t *) scan_arr_mem_18839)[local_tid_18833];\n            ((__global int32_t *) mem_18345)[gtid_17959] = ((__local\n                                                             int32_t *) scan_arr_mem_18841)[local_tid_18833];\n            ((__global int32_t *) mem_18348)[gtid_17959] = ((__local\n                                                             int32_t *) scan_arr_mem_18843)[local_tid_18833];\n            ((__global int32_t *) mem_18351)[gtid_17959] = ((__local\n                                 ",
                   "                            int32_t *) scan_arr_mem_18845)[local_tid_18833];\n            ((__global int32_t *) mem_18354)[gtid_17959] = ((__local\n                                                             int32_t *) scan_arr_mem_18847)[local_tid_18833];\n        }\n    }\n}\n__kernel void scan_stage2_17969(__local volatile\n                                int64_t *scan_arr_mem_18914_backing_aligned_0,\n                                int32_t sizze_17490, int32_t num_groups_17966,\n                                __global unsigned char *mem_18358,\n                                int32_t num_threads_18884)\n{\n    const int32_t segscan_group_sizze_17964 = mainzisegscan_group_sizze_17963;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_18914_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_18914_backing_aligned_0;\n    int32_t global_tid_18909;\n    int32_t local_tid_18910;\n    int32_t group_sizze_18913;\n    int32_t wave_sizze_18912;\n    int32_t group_tid_18911;\n    \n    global_tid_18909 = get_global_id(0);\n    local_tid_18910 = get_local_id(0);\n    group_sizze_18913 = get_local_size(0);\n    wave_sizze_18912 = LOCKSTEP_WIDTH;\n    group_tid_18911 = get_group_id(0);\n    \n    int32_t phys_tid_17969 = global_tid_18909;\n    __local char *scan_arr_mem_18914;\n    \n    scan_arr_mem_18914 = (__local char *) scan_arr_mem_18914_backing_0;\n    \n    int32_t flat_idx_18916 = (local_tid_18910 + 1) *\n            (segscan_group_sizze_17964 * squot32(sizze_17490 +\n                                                 num_threads_18884 - 1,\n                                                 num_threads_18884)) - 1;\n    int32_t gtid_17968 = flat_idx_18916;\n    \n    // threads in bound read carries; others get neutral element\n    {\n        if (slt32(gtid_17968, sizze_17490)) {\n            ((__local int32_t *) scan_arr_mem_18914)[local_tid_18910] =\n              ",
                   "  ((__global int32_t *) mem_18358)[gtid_17968];\n        } else {\n            ((__local int32_t *) scan_arr_mem_18914)[local_tid_18910] = 0;\n        }\n    }\n    \n    int32_t x_18903;\n    int32_t x_18904;\n    int32_t x_18917;\n    int32_t x_18918;\n    int32_t skip_threads_18920;\n    \n    if (slt32(local_tid_18910, num_groups_17966)) {\n        x_18904 = ((volatile __local\n                    int32_t *) scan_arr_mem_18914)[local_tid_18910];\n    }\n    // in-block scan (hopefully no barriers needed)\n    {\n        skip_threads_18920 = 1;\n        while (slt32(skip_threads_18920, 32)) {\n            if (sle32(skip_threads_18920, local_tid_18910 -\n                      squot32(local_tid_18910, 32) * 32) &&\n                slt32(local_tid_18910, num_groups_17966)) {\n                // read operands\n                {\n                    x_18903 = ((volatile __local\n                                int32_t *) scan_arr_mem_18914)[local_tid_18910 -\n                                                               skip_threads_18920];\n                }\n                // perform operation\n                {\n                    int32_t res_18905 = x_18903 + x_18904;\n                    \n                    x_18904 = res_18905;\n                }\n            }\n            if (sle32(wave_sizze_18912, skip_threads_18920)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (sle32(skip_threads_18920, local_tid_18910 -\n                      squot32(local_tid_18910, 32) * 32) &&\n                slt32(local_tid_18910, num_groups_17966)) {\n                // write result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18914)[local_tid_18910] = x_18904;\n                }\n            }\n            if (sle32(wave_sizze_18912, skip_threads_18920)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_threads_18920 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i",
                   "' writes its result to offset 'i'\n    {\n        if ((local_tid_18910 - squot32(local_tid_18910, 32) * 32) == 31 &&\n            slt32(local_tid_18910, num_groups_17966)) {\n            ((volatile __local\n              int32_t *) scan_arr_mem_18914)[squot32(local_tid_18910, 32)] =\n                x_18904;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n    {\n        int32_t skip_threads_18921;\n        \n        if (squot32(local_tid_18910, 32) == 0 && slt32(local_tid_18910,\n                                                       num_groups_17966)) {\n            x_18918 = ((volatile __local\n                        int32_t *) scan_arr_mem_18914)[local_tid_18910];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_18921 = 1;\n            while (slt32(skip_threads_18921, 32)) {\n                if (sle32(skip_threads_18921, local_tid_18910 -\n                          squot32(local_tid_18910, 32) * 32) &&\n                    (squot32(local_tid_18910, 32) == 0 && slt32(local_tid_18910,\n                                                                num_groups_17966))) {\n                    // read operands\n                    {\n                        x_18917 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18914)[local_tid_18910 -\n                                                                   skip_threads_18921];\n                    }\n                    // perform operation\n                    {\n                        int32_t res_18919 = x_18917 + x_18918;\n                        \n                        x_18918 = res_18919;\n                    }\n                }\n                if (sle32(wave_sizze_18912, skip_threads_18921)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_18921, local_tid_18910 -\n                          squot32(local_tid_189",
                   "10, 32) * 32) &&\n                    (squot32(local_tid_18910, 32) == 0 && slt32(local_tid_18910,\n                                                                num_groups_17966))) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18914)[local_tid_18910] =\n                            x_18918;\n                    }\n                }\n                if (sle32(wave_sizze_18912, skip_threads_18921)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_18921 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_18910, 32) == 0 || !slt32(local_tid_18910,\n                                                          num_groups_17966))) {\n            // read operands\n            {\n                x_18903 = ((volatile __local\n                            int32_t *) scan_arr_mem_18914)[squot32(local_tid_18910,\n                                                                   32) - 1];\n            }\n            // perform operation\n            {\n                int32_t res_18905 = x_18903 + x_18904;\n                \n                x_18904 = res_18905;\n            }\n            // write final result\n            {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18914)[local_tid_18910] = x_18904;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_18910, 32) == 0) {\n            ((volatile __local int32_t *) scan_arr_mem_18914)[local_tid_18910] =\n                x_18904;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // threads in bounds write scanned carries\n    {\n        if (slt32(gtid_17968, sizze_17490)) {\n            ((__global int32_t *) mem_18358)[gtid_17968] = ((__local\n                              ",
                   "                               int32_t *) scan_arr_mem_18914)[local_tid_18910];\n        }\n    }\n}\n__kernel void scan_stage2_18044(__local volatile\n                                int64_t *scan_arr_mem_18988_backing_aligned_0,\n                                __local volatile\n                                int64_t *scan_arr_mem_18990_backing_aligned_1,\n                                int32_t sizze_17490, int32_t num_groups_18041,\n                                __global unsigned char *mem_18368, __global\n                                unsigned char *mem_18371,\n                                int32_t num_threads_18944)\n{\n    const int32_t segscan_group_sizze_18039 = mainzisegscan_group_sizze_18038;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_18988_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_18988_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_18990_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_18990_backing_aligned_1;\n    int32_t global_tid_18983;\n    int32_t local_tid_18984;\n    int32_t group_sizze_18987;\n    int32_t wave_sizze_18986;\n    int32_t group_tid_18985;\n    \n    global_tid_18983 = get_global_id(0);\n    local_tid_18984 = get_local_id(0);\n    group_sizze_18987 = get_local_size(0);\n    wave_sizze_18986 = LOCKSTEP_WIDTH;\n    group_tid_18985 = get_group_id(0);\n    \n    int32_t phys_tid_18044 = global_tid_18983;\n    __local char *scan_arr_mem_18988;\n    \n    scan_arr_mem_18988 = (__local char *) scan_arr_mem_18988_backing_0;\n    \n    __local char *scan_arr_mem_18990;\n    \n    scan_arr_mem_18990 = (__local char *) scan_arr_mem_18990_backing_1;\n    \n    int32_t flat_idx_18992 = (local_tid_18984 + 1) *\n            (segscan_group_sizze_18039 * squot32(sizze_17490 +\n                                                 num_threads_18944 - 1,\n",
                   "                                                 num_threads_18944)) - 1;\n    int32_t gtid_18043 = flat_idx_18992;\n    \n    // threads in bound read carries; others get neutral element\n    {\n        if (slt32(gtid_18043, sizze_17490)) {\n            ((__local int32_t *) scan_arr_mem_18988)[local_tid_18984] =\n                ((__global int32_t *) mem_18368)[gtid_18043];\n            ((__local int32_t *) scan_arr_mem_18990)[local_tid_18984] =\n                ((__global int32_t *) mem_18371)[gtid_18043];\n        } else {\n            ((__local int32_t *) scan_arr_mem_18988)[local_tid_18984] = 0;\n            ((__local int32_t *) scan_arr_mem_18990)[local_tid_18984] = 0;\n        }\n    }\n    \n    int32_t x_18971;\n    int32_t x_18972;\n    int32_t x_18973;\n    int32_t x_18974;\n    int32_t x_18993;\n    int32_t x_18994;\n    int32_t x_18995;\n    int32_t x_18996;\n    int32_t skip_threads_18999;\n    \n    if (slt32(local_tid_18984, num_groups_18041)) {\n        x_18973 = ((volatile __local\n                    int32_t *) scan_arr_mem_18988)[local_tid_18984];\n        x_18974 = ((volatile __local\n                    int32_t *) scan_arr_mem_18990)[local_tid_18984];\n    }\n    // in-block scan (hopefully no barriers needed)\n    {\n        skip_threads_18999 = 1;\n        while (slt32(skip_threads_18999, 32)) {\n            if (sle32(skip_threads_18999, local_tid_18984 -\n                      squot32(local_tid_18984, 32) * 32) &&\n                slt32(local_tid_18984, num_groups_18041)) {\n                // read operands\n                {\n                    x_18971 = ((volatile __local\n                                int32_t *) scan_arr_mem_18988)[local_tid_18984 -\n                                                               skip_threads_18999];\n                    x_18972 = ((volatile __local\n                                int32_t *) scan_arr_mem_18990)[local_tid_18984 -\n                                                               skip_threads_18999];\n                }\n                /",
                   "/ perform operation\n                {\n                    int32_t res_18975 = x_18971 + x_18973;\n                    int32_t res_18976 = x_18972 + x_18974;\n                    \n                    x_18973 = res_18975;\n                    x_18974 = res_18976;\n                }\n            }\n            if (sle32(wave_sizze_18986, skip_threads_18999)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (sle32(skip_threads_18999, local_tid_18984 -\n                      squot32(local_tid_18984, 32) * 32) &&\n                slt32(local_tid_18984, num_groups_18041)) {\n                // write result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18988)[local_tid_18984] = x_18973;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_18990)[local_tid_18984] = x_18974;\n                }\n            }\n            if (sle32(wave_sizze_18986, skip_threads_18999)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_threads_18999 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i' writes its result to offset 'i'\n    {\n        if ((local_tid_18984 - squot32(local_tid_18984, 32) * 32) == 31 &&\n            slt32(local_tid_18984, num_groups_18041)) {\n            ((volatile __local\n              int32_t *) scan_arr_mem_18988)[squot32(local_tid_18984, 32)] =\n                x_18973;\n            ((volatile __local\n              int32_t *) scan_arr_mem_18990)[squot32(local_tid_18984, 32)] =\n                x_18974;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n    {\n        int32_t skip_threads_19000;\n        \n        if (squot32(local_tid_18984, 32) == 0 && slt32(local_tid_18984,\n                                                       num_groups_18041)) {\n            x_18995 = ((volatile __local\n                        int32_t ",
                   "*) scan_arr_mem_18988)[local_tid_18984];\n            x_18996 = ((volatile __local\n                        int32_t *) scan_arr_mem_18990)[local_tid_18984];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_19000 = 1;\n            while (slt32(skip_threads_19000, 32)) {\n                if (sle32(skip_threads_19000, local_tid_18984 -\n                          squot32(local_tid_18984, 32) * 32) &&\n                    (squot32(local_tid_18984, 32) == 0 && slt32(local_tid_18984,\n                                                                num_groups_18041))) {\n                    // read operands\n                    {\n                        x_18993 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18988)[local_tid_18984 -\n                                                                   skip_threads_19000];\n                        x_18994 = ((volatile __local\n                                    int32_t *) scan_arr_mem_18990)[local_tid_18984 -\n                                                                   skip_threads_19000];\n                    }\n                    // perform operation\n                    {\n                        int32_t res_18997 = x_18993 + x_18995;\n                        int32_t res_18998 = x_18994 + x_18996;\n                        \n                        x_18995 = res_18997;\n                        x_18996 = res_18998;\n                    }\n                }\n                if (sle32(wave_sizze_18986, skip_threads_19000)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_19000, local_tid_18984 -\n                          squot32(local_tid_18984, 32) * 32) &&\n                    (squot32(local_tid_18984, 32) == 0 && slt32(local_tid_18984,\n                                                                num_groups_18041))) {\n                    // write result\n                    {\n                     ",
                   "   ((volatile __local\n                          int32_t *) scan_arr_mem_18988)[local_tid_18984] =\n                            x_18995;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_18990)[local_tid_18984] =\n                            x_18996;\n                    }\n                }\n                if (sle32(wave_sizze_18986, skip_threads_19000)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_19000 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_18984, 32) == 0 || !slt32(local_tid_18984,\n                                                          num_groups_18041))) {\n            // read operands\n            {\n                x_18971 = ((volatile __local\n                            int32_t *) scan_arr_mem_18988)[squot32(local_tid_18984,\n                                                                   32) - 1];\n                x_18972 = ((volatile __local\n                            int32_t *) scan_arr_mem_18990)[squot32(local_tid_18984,\n                                                                   32) - 1];\n            }\n            // perform operation\n            {\n                int32_t res_18975 = x_18971 + x_18973;\n                int32_t res_18976 = x_18972 + x_18974;\n                \n                x_18973 = res_18975;\n                x_18974 = res_18976;\n            }\n            // write final result\n            {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18988)[local_tid_18984] = x_18973;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_18990)[local_tid_18984] = x_18974;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_18984, 32) == 0) {\n            ((volatile __local int32_t *) scan_arr_",
                   "mem_18988)[local_tid_18984] =\n                x_18973;\n            ((volatile __local int32_t *) scan_arr_mem_18990)[local_tid_18984] =\n                x_18974;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // threads in bounds write scanned carries\n    {\n        if (slt32(gtid_18043, sizze_17490)) {\n            ((__global int32_t *) mem_18368)[gtid_18043] = ((__local\n                                                             int32_t *) scan_arr_mem_18988)[local_tid_18984];\n            ((__global int32_t *) mem_18371)[gtid_18043] = ((__local\n                                                             int32_t *) scan_arr_mem_18990)[local_tid_18984];\n        }\n    }\n}\n__kernel void scan_stage2_18065(__local volatile\n                                int64_t *scan_arr_mem_19070_backing_aligned_0,\n                                __local volatile\n                                int64_t *scan_arr_mem_19072_backing_aligned_1,\n                                int32_t aoa_len_17711, int32_t num_groups_18062,\n                                __global unsigned char *mem_18378, __global\n                                unsigned char *mem_18381,\n                                int32_t num_threads_19014)\n{\n    const int32_t segscan_group_sizze_18060 = mainzisegscan_group_sizze_18059;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_19070_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_19070_backing_aligned_0;\n    __local volatile char *restrict scan_arr_mem_19072_backing_1 =\n                          (__local volatile\n                           char *) scan_arr_mem_19072_backing_aligned_1;\n    int32_t global_tid_19065;\n    int32_t local_tid_19066;\n    int32_t group_sizze_19069;\n    int32_t wave_sizze_19068;\n    int32_t group_tid_19067;\n    \n    global_tid_19065 = get_global_id(0);\n    local_tid_19066 = get_local_id(0);\n    group_s",
                   "izze_19069 = get_local_size(0);\n    wave_sizze_19068 = LOCKSTEP_WIDTH;\n    group_tid_19067 = get_group_id(0);\n    \n    int32_t phys_tid_18065 = global_tid_19065;\n    __local char *scan_arr_mem_19070;\n    \n    scan_arr_mem_19070 = (__local char *) scan_arr_mem_19070_backing_0;\n    \n    __local char *scan_arr_mem_19072;\n    \n    scan_arr_mem_19072 = (__local char *) scan_arr_mem_19072_backing_1;\n    \n    int32_t flat_idx_19074 = (local_tid_19066 + 1) *\n            (segscan_group_sizze_18060 * squot32(aoa_len_17711 +\n                                                 num_threads_19014 - 1,\n                                                 num_threads_19014)) - 1;\n    int32_t gtid_18064 = flat_idx_19074;\n    \n    // threads in bound read carries; others get neutral element\n    {\n        if (slt32(gtid_18064, aoa_len_17711)) {\n            ((__local int32_t *) scan_arr_mem_19070)[local_tid_19066] =\n                ((__global int32_t *) mem_18378)[gtid_18064];\n            ((__local int32_t *) scan_arr_mem_19072)[local_tid_19066] =\n                ((__global int32_t *) mem_18381)[gtid_18064];\n        } else {\n            ((__local int32_t *) scan_arr_mem_19070)[local_tid_19066] = 0;\n            ((__local int32_t *) scan_arr_mem_19072)[local_tid_19066] = 0;\n        }\n    }\n    \n    int32_t x_19047;\n    int32_t x_19048;\n    int32_t x_19049;\n    int32_t x_19050;\n    int32_t x_19075;\n    int32_t x_19076;\n    int32_t x_19077;\n    int32_t x_19078;\n    int32_t skip_threads_19084;\n    \n    if (slt32(local_tid_19066, num_groups_18062)) {\n        x_19049 = ((volatile __local\n                    int32_t *) scan_arr_mem_19070)[local_tid_19066];\n        x_19050 = ((volatile __local\n                    int32_t *) scan_arr_mem_19072)[local_tid_19066];\n    }\n    // in-block scan (hopefully no barriers needed)\n    {\n        skip_threads_19084 = 1;\n        while (slt32(skip_threads_19084, 32)) {\n            if (sle32(skip_threads_19084, local_tid_19066 -\n                      squot32(local_tid_",
                   "19066, 32) * 32) &&\n                slt32(local_tid_19066, num_groups_18062)) {\n                // read operands\n                {\n                    x_19047 = ((volatile __local\n                                int32_t *) scan_arr_mem_19070)[local_tid_19066 -\n                                                               skip_threads_19084];\n                    x_19048 = ((volatile __local\n                                int32_t *) scan_arr_mem_19072)[local_tid_19066 -\n                                                               skip_threads_19084];\n                }\n                // perform operation\n                {\n                    int32_t f_19051 = x_19047 | x_19049;\n                    bool cond_19052 = x_19049 == 0;\n                    bool cond_19053 = !cond_19052;\n                    int32_t res_19054;\n                    \n                    if (cond_19053) {\n                        res_19054 = x_19050;\n                    } else {\n                        int32_t res_19055 = x_19048 + x_19050;\n                        \n                        res_19054 = res_19055;\n                    }\n                    x_19049 = f_19051;\n                    x_19050 = res_19054;\n                }\n            }\n            if (sle32(wave_sizze_19068, skip_threads_19084)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (sle32(skip_threads_19084, local_tid_19066 -\n                      squot32(local_tid_19066, 32) * 32) &&\n                slt32(local_tid_19066, num_groups_18062)) {\n                // write result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_19070)[local_tid_19066] = x_19049;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_19072)[local_tid_19066] = x_19050;\n                }\n            }\n            if (sle32(wave_sizze_19068, skip_threads_19084)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_th",
                   "reads_19084 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i' writes its result to offset 'i'\n    {\n        if ((local_tid_19066 - squot32(local_tid_19066, 32) * 32) == 31 &&\n            slt32(local_tid_19066, num_groups_18062)) {\n            ((volatile __local\n              int32_t *) scan_arr_mem_19070)[squot32(local_tid_19066, 32)] =\n                x_19049;\n            ((volatile __local\n              int32_t *) scan_arr_mem_19072)[squot32(local_tid_19066, 32)] =\n                x_19050;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n    {\n        int32_t skip_threads_19085;\n        \n        if (squot32(local_tid_19066, 32) == 0 && slt32(local_tid_19066,\n                                                       num_groups_18062)) {\n            x_19077 = ((volatile __local\n                        int32_t *) scan_arr_mem_19070)[local_tid_19066];\n            x_19078 = ((volatile __local\n                        int32_t *) scan_arr_mem_19072)[local_tid_19066];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_19085 = 1;\n            while (slt32(skip_threads_19085, 32)) {\n                if (sle32(skip_threads_19085, local_tid_19066 -\n                          squot32(local_tid_19066, 32) * 32) &&\n                    (squot32(local_tid_19066, 32) == 0 && slt32(local_tid_19066,\n                                                                num_groups_18062))) {\n                    // read operands\n                    {\n                        x_19075 = ((volatile __local\n                                    int32_t *) scan_arr_mem_19070)[local_tid_19066 -\n                                                                   skip_threads_19085];\n                        x_19076 = ((volatile __local\n                                    int32_t *) scan_arr_mem_19072)[local_tid_19066 -\n                     ",
                   "                                              skip_threads_19085];\n                    }\n                    // perform operation\n                    {\n                        int32_t f_19079 = x_19075 | x_19077;\n                        bool cond_19080 = x_19077 == 0;\n                        bool cond_19081 = !cond_19080;\n                        int32_t res_19082;\n                        \n                        if (cond_19081) {\n                            res_19082 = x_19078;\n                        } else {\n                            int32_t res_19083 = x_19076 + x_19078;\n                            \n                            res_19082 = res_19083;\n                        }\n                        x_19077 = f_19079;\n                        x_19078 = res_19082;\n                    }\n                }\n                if (sle32(wave_sizze_19068, skip_threads_19085)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_19085, local_tid_19066 -\n                          squot32(local_tid_19066, 32) * 32) &&\n                    (squot32(local_tid_19066, 32) == 0 && slt32(local_tid_19066,\n                                                                num_groups_18062))) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_19070)[local_tid_19066] =\n                            x_19077;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_19072)[local_tid_19066] =\n                            x_19078;\n                    }\n                }\n                if (sle32(wave_sizze_19068, skip_threads_19085)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_19085 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_19066, 32) == 0 ",
                   "|| !slt32(local_tid_19066,\n                                                          num_groups_18062))) {\n            // read operands\n            {\n                x_19047 = ((volatile __local\n                            int32_t *) scan_arr_mem_19070)[squot32(local_tid_19066,\n                                                                   32) - 1];\n                x_19048 = ((volatile __local\n                            int32_t *) scan_arr_mem_19072)[squot32(local_tid_19066,\n                                                                   32) - 1];\n            }\n            // perform operation\n            {\n                int32_t f_19051 = x_19047 | x_19049;\n                bool cond_19052 = x_19049 == 0;\n                bool cond_19053 = !cond_19052;\n                int32_t res_19054;\n                \n                if (cond_19053) {\n                    res_19054 = x_19050;\n                } else {\n                    int32_t res_19055 = x_19048 + x_19050;\n                    \n                    res_19054 = res_19055;\n                }\n                x_19049 = f_19051;\n                x_19050 = res_19054;\n            }\n            // write final result\n            {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_19070)[local_tid_19066] = x_19049;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_19072)[local_tid_19066] = x_19050;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_19066, 32) == 0) {\n            ((volatile __local int32_t *) scan_arr_mem_19070)[local_tid_19066] =\n                x_19049;\n            ((volatile __local int32_t *) scan_arr_mem_19072)[local_tid_19066] =\n                x_19050;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // threads in bounds write scanned carries\n    {\n        if (slt32(gtid_18064, aoa_len_17711)) {\n            ((__global int32_t *) mem_18378)[gtid_",
                   "18064] = ((__local\n                                                             int32_t *) scan_arr_mem_19070)[local_tid_19066];\n            ((__global int32_t *) mem_18381)[gtid_18064] = ((__local\n                                                             int32_t *) scan_arr_mem_19072)[local_tid_19066];\n        }\n    }\n}\n__kernel void scan_stage2_18225(__local volatile\n                                int64_t *scan_arr_mem_19147_backing_aligned_0,\n                                int32_t num_groups_18222,\n                                int32_t convop_x_18393, __global\n                                unsigned char *mem_18403,\n                                int32_t num_threads_19117)\n{\n    const int32_t segscan_group_sizze_18220 = mainzisegscan_group_sizze_18219;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict scan_arr_mem_19147_backing_0 =\n                          (__local volatile\n                           char *) scan_arr_mem_19147_backing_aligned_0;\n    int32_t global_tid_19142;\n    int32_t local_tid_19143;\n    int32_t group_sizze_19146;\n    int32_t wave_sizze_19145;\n    int32_t group_tid_19144;\n    \n    global_tid_19142 = get_global_id(0);\n    local_tid_19143 = get_local_id(0);\n    group_sizze_19146 = get_local_size(0);\n    wave_sizze_19145 = LOCKSTEP_WIDTH;\n    group_tid_19144 = get_group_id(0);\n    \n    int32_t phys_tid_18225 = global_tid_19142;\n    __local char *scan_arr_mem_19147;\n    \n    scan_arr_mem_19147 = (__local char *) scan_arr_mem_19147_backing_0;\n    \n    int32_t flat_idx_19149 = (local_tid_19143 + 1) *\n            (segscan_group_sizze_18220 * squot32(convop_x_18393 +\n                                                 num_threads_19117 - 1,\n                                                 num_threads_19117)) - 1;\n    int32_t gtid_18224 = flat_idx_19149;\n    \n    // threads in bound read carries; others get neutral element\n    {\n        if (slt32(gtid_18224, convop_x_183",
                   "93)) {\n            ((__local int32_t *) scan_arr_mem_19147)[local_tid_19143] =\n                ((__global int32_t *) mem_18403)[gtid_18224];\n        } else {\n            ((__local int32_t *) scan_arr_mem_19147)[local_tid_19143] = 0;\n        }\n    }\n    \n    int32_t x_19136;\n    int32_t y_19137;\n    int32_t x_19150;\n    int32_t y_19151;\n    int32_t skip_threads_19153;\n    \n    if (slt32(local_tid_19143, num_groups_18222)) {\n        y_19137 = ((volatile __local\n                    int32_t *) scan_arr_mem_19147)[local_tid_19143];\n    }\n    // in-block scan (hopefully no barriers needed)\n    {\n        skip_threads_19153 = 1;\n        while (slt32(skip_threads_19153, 32)) {\n            if (sle32(skip_threads_19153, local_tid_19143 -\n                      squot32(local_tid_19143, 32) * 32) &&\n                slt32(local_tid_19143, num_groups_18222)) {\n                // read operands\n                {\n                    x_19136 = ((volatile __local\n                                int32_t *) scan_arr_mem_19147)[local_tid_19143 -\n                                                               skip_threads_19153];\n                }\n                // perform operation\n                {\n                    int32_t zz_19138 = x_19136 + y_19137;\n                    \n                    y_19137 = zz_19138;\n                }\n            }\n            if (sle32(wave_sizze_19145, skip_threads_19153)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (sle32(skip_threads_19153, local_tid_19143 -\n                      squot32(local_tid_19143, 32) * 32) &&\n                slt32(local_tid_19143, num_groups_18222)) {\n                // write result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_19147)[local_tid_19143] = y_19137;\n                }\n            }\n            if (sle32(wave_sizze_19145, skip_threads_19153)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_threads_1",
                   "9153 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i' writes its result to offset 'i'\n    {\n        if ((local_tid_19143 - squot32(local_tid_19143, 32) * 32) == 31 &&\n            slt32(local_tid_19143, num_groups_18222)) {\n            ((volatile __local\n              int32_t *) scan_arr_mem_19147)[squot32(local_tid_19143, 32)] =\n                y_19137;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for warp 'i+1'\n    {\n        int32_t skip_threads_19154;\n        \n        if (squot32(local_tid_19143, 32) == 0 && slt32(local_tid_19143,\n                                                       num_groups_18222)) {\n            y_19151 = ((volatile __local\n                        int32_t *) scan_arr_mem_19147)[local_tid_19143];\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_19154 = 1;\n            while (slt32(skip_threads_19154, 32)) {\n                if (sle32(skip_threads_19154, local_tid_19143 -\n                          squot32(local_tid_19143, 32) * 32) &&\n                    (squot32(local_tid_19143, 32) == 0 && slt32(local_tid_19143,\n                                                                num_groups_18222))) {\n                    // read operands\n                    {\n                        x_19150 = ((volatile __local\n                                    int32_t *) scan_arr_mem_19147)[local_tid_19143 -\n                                                                   skip_threads_19154];\n                    }\n                    // perform operation\n                    {\n                        int32_t zz_19152 = x_19150 + y_19151;\n                        \n                        y_19151 = zz_19152;\n                    }\n                }\n                if (sle32(wave_sizze_19145, skip_threads_19154)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sl",
                   "e32(skip_threads_19154, local_tid_19143 -\n                          squot32(local_tid_19143, 32) * 32) &&\n                    (squot32(local_tid_19143, 32) == 0 && slt32(local_tid_19143,\n                                                                num_groups_18222))) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_19147)[local_tid_19143] =\n                            y_19151;\n                    }\n                }\n                if (sle32(wave_sizze_19145, skip_threads_19154)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_19154 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_19143, 32) == 0 || !slt32(local_tid_19143,\n                                                          num_groups_18222))) {\n            // read operands\n            {\n                x_19136 = ((volatile __local\n                            int32_t *) scan_arr_mem_19147)[squot32(local_tid_19143,\n                                                                   32) - 1];\n            }\n            // perform operation\n            {\n                int32_t zz_19138 = x_19136 + y_19137;\n                \n                y_19137 = zz_19138;\n            }\n            // write final result\n            {\n                ((volatile __local\n                  int32_t *) scan_arr_mem_19147)[local_tid_19143] = y_19137;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_19143, 32) == 0) {\n            ((volatile __local int32_t *) scan_arr_mem_19147)[local_tid_19143] =\n                y_19137;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // threads in bounds write scanned carries\n    {\n        if (slt32(gtid_18224, convop_x_18393)) {\n         ",
                   "   ((__global int32_t *) mem_18403)[gtid_18224] = ((__local\n                                                             int32_t *) scan_arr_mem_19147)[local_tid_19143];\n        }\n    }\n}\n__kernel void scan_stage3_18569(int32_t sizze_17490, __global\n                                unsigned char *mem_18307, __global\n                                unsigned char *mem_18310,\n                                int32_t num_threads_18512)\n{\n    const int32_t segscan_group_sizze_17836 = mainzisegscan_group_sizze_17835;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t scan_stage3_gtid_18569;\n    int32_t scan_stage3_ltid_18570;\n    int32_t scan_stage3_gid_18571;\n    \n    scan_stage3_gtid_18569 = get_global_id(0);\n    scan_stage3_ltid_18570 = get_local_id(0);\n    scan_stage3_gid_18571 = get_group_id(0);\n    \n    int32_t phys_tid_17841 = scan_stage3_gtid_18569;\n    int32_t gtid_17840 = scan_stage3_gtid_18569;\n    int32_t orig_group_18574 = squot32(scan_stage3_gtid_18569,\n                                       segscan_group_sizze_17836 *\n                                       squot32(sizze_17490 + num_threads_18512 -\n                                               1, num_threads_18512));\n    int32_t carry_in_flat_idx_18575 = orig_group_18574 *\n            (segscan_group_sizze_17836 * squot32(sizze_17490 +\n                                                 num_threads_18512 - 1,\n                                                 num_threads_18512)) - 1;\n    \n    if (slt32(scan_stage3_gtid_18569, sizze_17490)) {\n        if (!(orig_group_18574 == 0 || scan_stage3_gtid_18569 ==\n              (orig_group_18574 + 1) * (segscan_group_sizze_17836 *\n                                        squot32(sizze_17490 +\n                                                num_threads_18512 - 1,\n                                                num_threads_18512)) - 1)) {\n            int32_t x_18545;\n            int32_t x_18546;\n            int32_t x_18547;\n  ",
                   "          int32_t x_18548;\n            \n            x_18545 = ((__global int32_t *) mem_18307)[carry_in_flat_idx_18575];\n            x_18546 = ((__global int32_t *) mem_18310)[carry_in_flat_idx_18575];\n            x_18547 = ((__global int32_t *) mem_18307)[gtid_17840];\n            x_18548 = ((__global int32_t *) mem_18310)[gtid_17840];\n            \n            int32_t res_18549 = x_18545 + x_18547;\n            int32_t res_18550 = x_18546 + x_18548;\n            \n            x_18545 = res_18549;\n            x_18546 = res_18550;\n            ((__global int32_t *) mem_18307)[gtid_17840] = x_18545;\n            ((__global int32_t *) mem_18310)[gtid_17840] = x_18546;\n        }\n    }\n}\n__kernel void scan_stage3_18649(int32_t aoa_len_17514, __global\n                                unsigned char *mem_18317, __global\n                                unsigned char *mem_18320,\n                                int32_t num_threads_18582)\n{\n    const int32_t segscan_group_sizze_17857 = mainzisegscan_group_sizze_17856;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t scan_stage3_gtid_18649;\n    int32_t scan_stage3_ltid_18650;\n    int32_t scan_stage3_gid_18651;\n    \n    scan_stage3_gtid_18649 = get_global_id(0);\n    scan_stage3_ltid_18650 = get_local_id(0);\n    scan_stage3_gid_18651 = get_group_id(0);\n    \n    int32_t phys_tid_17862 = scan_stage3_gtid_18649;\n    int32_t gtid_17861 = scan_stage3_gtid_18649;\n    int32_t orig_group_18654 = squot32(scan_stage3_gtid_18649,\n                                       segscan_group_sizze_17857 *\n                                       squot32(aoa_len_17514 +\n                                               num_threads_18582 - 1,\n                                               num_threads_18582));\n    int32_t carry_in_flat_idx_18655 = orig_group_18654 *\n            (segscan_group_sizze_17857 * squot32(aoa_len_17514 +\n                                                 num_threads_18582 - 1,\n             ",
                   "                                    num_threads_18582)) - 1;\n    \n    if (slt32(scan_stage3_gtid_18649, aoa_len_17514)) {\n        if (!(orig_group_18654 == 0 || scan_stage3_gtid_18649 ==\n              (orig_group_18654 + 1) * (segscan_group_sizze_17857 *\n                                        squot32(aoa_len_17514 +\n                                                num_threads_18582 - 1,\n                                                num_threads_18582)) - 1)) {\n            int32_t x_18621;\n            int32_t x_18622;\n            int32_t x_18623;\n            int32_t x_18624;\n            \n            x_18621 = ((__global int32_t *) mem_18317)[carry_in_flat_idx_18655];\n            x_18622 = ((__global int32_t *) mem_18320)[carry_in_flat_idx_18655];\n            x_18623 = ((__global int32_t *) mem_18317)[gtid_17861];\n            x_18624 = ((__global int32_t *) mem_18320)[gtid_17861];\n            \n            int32_t f_18625 = x_18621 | x_18623;\n            bool cond_18626 = slt32(0, x_18623);\n            int32_t res_18627;\n            \n            if (cond_18626) {\n                res_18627 = x_18624;\n            } else {\n                int32_t res_18628 = x_18622 + x_18624;\n                \n                res_18627 = res_18628;\n            }\n            x_18621 = f_18625;\n            x_18622 = res_18627;\n            ((__global int32_t *) mem_18317)[gtid_17861] = x_18621;\n            ((__global int32_t *) mem_18320)[gtid_17861] = x_18622;\n        }\n    }\n}\n__kernel void scan_stage3_18695(int32_t sizze_17490, __global\n                                unsigned char *mem_18324,\n                                int32_t num_threads_18657)\n{\n    const int32_t segscan_group_sizze_17866 = mainzisegscan_group_sizze_17865;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t scan_stage3_gtid_18695;\n    int32_t scan_stage3_ltid_18696;\n    int32_t scan_stage3_gid_18697;\n    \n    scan_stage3_gtid_18695 = get_global_id(0);\n    scan_sta",
                   "ge3_ltid_18696 = get_local_id(0);\n    scan_stage3_gid_18697 = get_group_id(0);\n    \n    int32_t phys_tid_17871 = scan_stage3_gtid_18695;\n    int32_t gtid_17870 = scan_stage3_gtid_18695;\n    int32_t orig_group_18700 = squot32(scan_stage3_gtid_18695,\n                                       segscan_group_sizze_17866 *\n                                       squot32(sizze_17490 + num_threads_18657 -\n                                               1, num_threads_18657));\n    int32_t carry_in_flat_idx_18701 = orig_group_18700 *\n            (segscan_group_sizze_17866 * squot32(sizze_17490 +\n                                                 num_threads_18657 - 1,\n                                                 num_threads_18657)) - 1;\n    \n    if (slt32(scan_stage3_gtid_18695, sizze_17490)) {\n        if (!(orig_group_18700 == 0 || scan_stage3_gtid_18695 ==\n              (orig_group_18700 + 1) * (segscan_group_sizze_17866 *\n                                        squot32(sizze_17490 +\n                                                num_threads_18657 - 1,\n                                                num_threads_18657)) - 1)) {\n            int32_t x_18679;\n            int32_t x_18680;\n            \n            x_18679 = ((__global int32_t *) mem_18324)[carry_in_flat_idx_18701];\n            x_18680 = ((__global int32_t *) mem_18324)[gtid_17870];\n            \n            int32_t res_18681 = x_18679 + x_18680;\n            \n            x_18679 = res_18681;\n            ((__global int32_t *) mem_18324)[gtid_17870] = x_18679;\n        }\n    }\n}\n__kernel void scan_stage3_18876(int32_t aoa_len_17582, __global\n                                unsigned char *mem_18339, __global\n                                unsigned char *mem_18342, __global\n                                unsigned char *mem_18345, __global\n                                unsigned char *mem_18348, __global\n                                unsigned char *mem_18351, __global\n                                unsigned char *mem",
                   "_18354,\n                                int32_t num_threads_18713)\n{\n    const int32_t segscan_group_sizze_17955 = mainzisegscan_group_sizze_17954;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t scan_stage3_gtid_18876;\n    int32_t scan_stage3_ltid_18877;\n    int32_t scan_stage3_gid_18878;\n    \n    scan_stage3_gtid_18876 = get_global_id(0);\n    scan_stage3_ltid_18877 = get_local_id(0);\n    scan_stage3_gid_18878 = get_group_id(0);\n    \n    int32_t phys_tid_17960 = scan_stage3_gtid_18876;\n    int32_t gtid_17959 = scan_stage3_gtid_18876;\n    int32_t orig_group_18881 = squot32(scan_stage3_gtid_18876,\n                                       segscan_group_sizze_17955 *\n                                       squot32(aoa_len_17582 +\n                                               num_threads_18713 - 1,\n                                               num_threads_18713));\n    int32_t carry_in_flat_idx_18882 = orig_group_18881 *\n            (segscan_group_sizze_17955 * squot32(aoa_len_17582 +\n                                                 num_threads_18713 - 1,\n                                                 num_threads_18713)) - 1;\n    \n    if (slt32(scan_stage3_gtid_18876, aoa_len_17582)) {\n        if (!(orig_group_18881 == 0 || scan_stage3_gtid_18876 ==\n              (orig_group_18881 + 1) * (segscan_group_sizze_17955 *\n                                        squot32(aoa_len_17582 +\n                                                num_threads_18713 - 1,\n                                                num_threads_18713)) - 1)) {\n            int32_t x_18808;\n            int32_t x_18809;\n            int32_t x_18810;\n            int32_t x_18811;\n            int32_t x_18812;\n            int32_t x_18813;\n            int32_t x_18814;\n            int32_t x_18815;\n            int32_t x_18816;\n            int32_t x_18817;\n            int32_t x_18818;\n            int32_t x_18819;\n            \n            x_18808 = ((__global int32_t",
                   " *) mem_18339)[carry_in_flat_idx_18882];\n            x_18809 = ((__global int32_t *) mem_18342)[carry_in_flat_idx_18882];\n            x_18810 = ((__global int32_t *) mem_18345)[carry_in_flat_idx_18882];\n            x_18811 = ((__global int32_t *) mem_18348)[carry_in_flat_idx_18882];\n            x_18812 = ((__global int32_t *) mem_18351)[carry_in_flat_idx_18882];\n            x_18813 = ((__global int32_t *) mem_18354)[carry_in_flat_idx_18882];\n            x_18814 = ((__global int32_t *) mem_18339)[gtid_17959];\n            x_18815 = ((__global int32_t *) mem_18342)[gtid_17959];\n            x_18816 = ((__global int32_t *) mem_18345)[gtid_17959];\n            x_18817 = ((__global int32_t *) mem_18348)[gtid_17959];\n            x_18818 = ((__global int32_t *) mem_18351)[gtid_17959];\n            x_18819 = ((__global int32_t *) mem_18354)[gtid_17959];\n            \n            int32_t f_18820 = x_18808 | x_18814;\n            bool cond_18821 = slt32(0, x_18814);\n            int32_t res_18822;\n            \n            if (cond_18821) {\n                res_18822 = x_18815;\n            } else {\n                int32_t res_18823 = x_18809 + x_18815;\n                \n                res_18822 = res_18823;\n            }\n            \n            int32_t f_18824 = x_18810 | x_18816;\n            bool cond_18825 = slt32(0, x_18816);\n            int32_t res_18826;\n            \n            if (cond_18825) {\n                res_18826 = x_18817;\n            } else {\n                int32_t res_18827 = x_18811 + x_18817;\n                \n                res_18826 = res_18827;\n            }\n            \n            int32_t f_18828 = x_18812 | x_18818;\n            bool cond_18829 = slt32(0, x_18818);\n            int32_t res_18830;\n            \n            if (cond_18829) {\n                res_18830 = x_18819;\n            } else {\n                int32_t res_18831 = x_18813 + x_18819;\n                \n                res_18830 = res_18831;\n            }\n            x_18808 = f_18820;\n           ",
                   " x_18809 = res_18822;\n            x_18810 = f_18824;\n            x_18811 = res_18826;\n            x_18812 = f_18828;\n            x_18813 = res_18830;\n            ((__global int32_t *) mem_18339)[gtid_17959] = x_18808;\n            ((__global int32_t *) mem_18342)[gtid_17959] = x_18809;\n            ((__global int32_t *) mem_18345)[gtid_17959] = x_18810;\n            ((__global int32_t *) mem_18348)[gtid_17959] = x_18811;\n            ((__global int32_t *) mem_18351)[gtid_17959] = x_18812;\n            ((__global int32_t *) mem_18354)[gtid_17959] = x_18813;\n        }\n    }\n}\n__kernel void scan_stage3_18922(int32_t sizze_17490, __global\n                                unsigned char *mem_18358,\n                                int32_t num_threads_18884)\n{\n    const int32_t segscan_group_sizze_17964 = mainzisegscan_group_sizze_17963;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t scan_stage3_gtid_18922;\n    int32_t scan_stage3_ltid_18923;\n    int32_t scan_stage3_gid_18924;\n    \n    scan_stage3_gtid_18922 = get_global_id(0);\n    scan_stage3_ltid_18923 = get_local_id(0);\n    scan_stage3_gid_18924 = get_group_id(0);\n    \n    int32_t phys_tid_17969 = scan_stage3_gtid_18922;\n    int32_t gtid_17968 = scan_stage3_gtid_18922;\n    int32_t orig_group_18927 = squot32(scan_stage3_gtid_18922,\n                                       segscan_group_sizze_17964 *\n                                       squot32(sizze_17490 + num_threads_18884 -\n                                               1, num_threads_18884));\n    int32_t carry_in_flat_idx_18928 = orig_group_18927 *\n            (segscan_group_sizze_17964 * squot32(sizze_17490 +\n                                                 num_threads_18884 - 1,\n                                                 num_threads_18884)) - 1;\n    \n    if (slt32(scan_stage3_gtid_18922, sizze_17490)) {\n        if (!(orig_group_18927 == 0 || scan_stage3_gtid_18922 ==\n              (orig_group_18927 + 1) * (segsc",
                   "an_group_sizze_17964 *\n                                        squot32(sizze_17490 +\n                                                num_threads_18884 - 1,\n                                                num_threads_18884)) - 1)) {\n            int32_t x_18906;\n            int32_t x_18907;\n            \n            x_18906 = ((__global int32_t *) mem_18358)[carry_in_flat_idx_18928];\n            x_18907 = ((__global int32_t *) mem_18358)[gtid_17968];\n            \n            int32_t res_18908 = x_18906 + x_18907;\n            \n            x_18906 = res_18908;\n            ((__global int32_t *) mem_18358)[gtid_17968] = x_18906;\n        }\n    }\n}\n__kernel void scan_stage3_19001(int32_t sizze_17490, __global\n                                unsigned char *mem_18368, __global\n                                unsigned char *mem_18371,\n                                int32_t num_threads_18944)\n{\n    const int32_t segscan_group_sizze_18039 = mainzisegscan_group_sizze_18038;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t scan_stage3_gtid_19001;\n    int32_t scan_stage3_ltid_19002;\n    int32_t scan_stage3_gid_19003;\n    \n    scan_stage3_gtid_19001 = get_global_id(0);\n    scan_stage3_ltid_19002 = get_local_id(0);\n    scan_stage3_gid_19003 = get_group_id(0);\n    \n    int32_t phys_tid_18044 = scan_stage3_gtid_19001;\n    int32_t gtid_18043 = scan_stage3_gtid_19001;\n    int32_t orig_group_19006 = squot32(scan_stage3_gtid_19001,\n                                       segscan_group_sizze_18039 *\n                                       squot32(sizze_17490 + num_threads_18944 -\n                                               1, num_threads_18944));\n    int32_t carry_in_flat_idx_19007 = orig_group_19006 *\n            (segscan_group_sizze_18039 * squot32(sizze_17490 +\n                                                 num_threads_18944 - 1,\n                                                 num_threads_18944)) - 1;\n    \n    if (slt32(scan_stage3",
                   "_gtid_19001, sizze_17490)) {\n        if (!(orig_group_19006 == 0 || scan_stage3_gtid_19001 ==\n              (orig_group_19006 + 1) * (segscan_group_sizze_18039 *\n                                        squot32(sizze_17490 +\n                                                num_threads_18944 - 1,\n                                                num_threads_18944)) - 1)) {\n            int32_t x_18977;\n            int32_t x_18978;\n            int32_t x_18979;\n            int32_t x_18980;\n            \n            x_18977 = ((__global int32_t *) mem_18368)[carry_in_flat_idx_19007];\n            x_18978 = ((__global int32_t *) mem_18371)[carry_in_flat_idx_19007];\n            x_18979 = ((__global int32_t *) mem_18368)[gtid_18043];\n            x_18980 = ((__global int32_t *) mem_18371)[gtid_18043];\n            \n            int32_t res_18981 = x_18977 + x_18979;\n            int32_t res_18982 = x_18978 + x_18980;\n            \n            x_18977 = res_18981;\n            x_18978 = res_18982;\n            ((__global int32_t *) mem_18368)[gtid_18043] = x_18977;\n            ((__global int32_t *) mem_18371)[gtid_18043] = x_18978;\n        }\n    }\n}\n__kernel void scan_stage3_19086(int32_t aoa_len_17711, __global\n                                unsigned char *mem_18378, __global\n                                unsigned char *mem_18381,\n                                int32_t num_threads_19014)\n{\n    const int32_t segscan_group_sizze_18060 = mainzisegscan_group_sizze_18059;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t scan_stage3_gtid_19086;\n    int32_t scan_stage3_ltid_19087;\n    int32_t scan_stage3_gid_19088;\n    \n    scan_stage3_gtid_19086 = get_global_id(0);\n    scan_stage3_ltid_19087 = get_local_id(0);\n    scan_stage3_gid_19088 = get_group_id(0);\n    \n    int32_t phys_tid_18065 = scan_stage3_gtid_19086;\n    int32_t gtid_18064 = scan_stage3_gtid_19086;\n    int32_t orig_group_19091 = squot32(scan_stage3_gtid_19086,\n               ",
                   "                        segscan_group_sizze_18060 *\n                                       squot32(aoa_len_17711 +\n                                               num_threads_19014 - 1,\n                                               num_threads_19014));\n    int32_t carry_in_flat_idx_19092 = orig_group_19091 *\n            (segscan_group_sizze_18060 * squot32(aoa_len_17711 +\n                                                 num_threads_19014 - 1,\n                                                 num_threads_19014)) - 1;\n    \n    if (slt32(scan_stage3_gtid_19086, aoa_len_17711)) {\n        if (!(orig_group_19091 == 0 || scan_stage3_gtid_19086 ==\n              (orig_group_19091 + 1) * (segscan_group_sizze_18060 *\n                                        squot32(aoa_len_17711 +\n                                                num_threads_19014 - 1,\n                                                num_threads_19014)) - 1)) {\n            int32_t x_19056;\n            int32_t x_19057;\n            int32_t x_19058;\n            int32_t x_19059;\n            \n            x_19056 = ((__global int32_t *) mem_18378)[carry_in_flat_idx_19092];\n            x_19057 = ((__global int32_t *) mem_18381)[carry_in_flat_idx_19092];\n            x_19058 = ((__global int32_t *) mem_18378)[gtid_18064];\n            x_19059 = ((__global int32_t *) mem_18381)[gtid_18064];\n            \n            int32_t f_19060 = x_19056 | x_19058;\n            bool cond_19061 = x_19058 == 0;\n            bool cond_19062 = !cond_19061;\n            int32_t res_19063;\n            \n            if (cond_19062) {\n                res_19063 = x_19059;\n            } else {\n                int32_t res_19064 = x_19057 + x_19059;\n                \n                res_19063 = res_19064;\n            }\n            x_19056 = f_19060;\n            x_19057 = res_19063;\n            ((__global int32_t *) mem_18378)[gtid_18064] = x_19056;\n            ((__global int32_t *) mem_18381)[gtid_18064] = x_19057;\n        }\n    }\n}\n__kernel void scan_stag",
                   "e3_19155(int32_t convop_x_18393, __global\n                                unsigned char *mem_18403,\n                                int32_t num_threads_19117)\n{\n    const int32_t segscan_group_sizze_18220 = mainzisegscan_group_sizze_18219;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t scan_stage3_gtid_19155;\n    int32_t scan_stage3_ltid_19156;\n    int32_t scan_stage3_gid_19157;\n    \n    scan_stage3_gtid_19155 = get_global_id(0);\n    scan_stage3_ltid_19156 = get_local_id(0);\n    scan_stage3_gid_19157 = get_group_id(0);\n    \n    int32_t phys_tid_18225 = scan_stage3_gtid_19155;\n    int32_t gtid_18224 = scan_stage3_gtid_19155;\n    int32_t orig_group_19160 = squot32(scan_stage3_gtid_19155,\n                                       segscan_group_sizze_18220 *\n                                       squot32(convop_x_18393 +\n                                               num_threads_19117 - 1,\n                                               num_threads_19117));\n    int32_t carry_in_flat_idx_19161 = orig_group_19160 *\n            (segscan_group_sizze_18220 * squot32(convop_x_18393 +\n                                                 num_threads_19117 - 1,\n                                                 num_threads_19117)) - 1;\n    \n    if (slt32(scan_stage3_gtid_19155, convop_x_18393)) {\n        if (!(orig_group_19160 == 0 || scan_stage3_gtid_19155 ==\n              (orig_group_19160 + 1) * (segscan_group_sizze_18220 *\n                                        squot32(convop_x_18393 +\n                                                num_threads_19117 - 1,\n                                                num_threads_19117)) - 1)) {\n            int32_t x_19139;\n            int32_t y_19140;\n            \n            x_19139 = ((__global int32_t *) mem_18403)[carry_in_flat_idx_19161];\n            y_19140 = ((__global int32_t *) mem_18403)[gtid_18224];\n            \n            int32_t zz_19141 = x_19139 + y_19140;\n            \n         ",
                   "   x_19139 = zz_19141;\n            ((__global int32_t *) mem_18403)[gtid_18224] = x_19139;\n        }\n    }\n}\n__kernel void segmap_17843(int32_t sizze_17490, int32_t aoa_len_17514, __global\n                           unsigned char *shp_mem_18302, __global\n                           unsigned char *mem_18310, __global\n                           unsigned char *mem_18313)\n{\n    const int32_t segmap_group_sizze_17847 = mainzisegmap_group_sizze_17846;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_18576;\n    int32_t local_tid_18577;\n    int32_t group_sizze_18580;\n    int32_t wave_sizze_18579;\n    int32_t group_tid_18578;\n    \n    global_tid_18576 = get_global_id(0);\n    local_tid_18577 = get_local_id(0);\n    group_sizze_18580 = get_local_size(0);\n    wave_sizze_18579 = LOCKSTEP_WIDTH;\n    group_tid_18578 = get_group_id(0);\n    \n    int32_t phys_tid_17843 = global_tid_18576;\n    int32_t write_i_17842 = group_tid_18578 * segmap_group_sizze_17847 +\n            local_tid_18577;\n    \n    if (slt32(write_i_17842, sizze_17490)) {\n        int32_t x_17518 = ((__global int32_t *) shp_mem_18302)[write_i_17842];\n        int32_t x_17519 = ((__global int32_t *) mem_18310)[write_i_17842];\n        int32_t res_17520 = 1 + write_i_17842;\n        bool cond_17521 = x_17518 == 0;\n        int32_t res_17522;\n        \n        if (cond_17521) {\n            res_17522 = -1;\n        } else {\n            res_17522 = x_17519;\n        }\n        if (sle32(0, res_17522) && slt32(res_17522, aoa_len_17514)) {\n            ((__global int32_t *) mem_18313)[res_17522] = res_17520;\n        }\n    }\n}\n__kernel void segmap_17873(int32_t sizze_17490, int32_t aoa_len_17582, __global\n                           unsigned char *shp_mem_18302, __global\n                           unsigned char *mem_18324, __global\n                           unsigned char *mem_18330)\n{\n    const int32_t segmap_group_sizze_17877 = mainzisegmap_group_sizze_17876;\n    const int ",
                   "block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_18702;\n    int32_t local_tid_18703;\n    int32_t group_sizze_18706;\n    int32_t wave_sizze_18705;\n    int32_t group_tid_18704;\n    \n    global_tid_18702 = get_global_id(0);\n    local_tid_18703 = get_local_id(0);\n    group_sizze_18706 = get_local_size(0);\n    wave_sizze_18705 = LOCKSTEP_WIDTH;\n    group_tid_18704 = get_group_id(0);\n    \n    int32_t phys_tid_17873 = global_tid_18702;\n    int32_t write_i_17872 = group_tid_18704 * segmap_group_sizze_17877 +\n            local_tid_18703;\n    \n    if (slt32(write_i_17872, sizze_17490)) {\n        int32_t x_17586 = ((__global int32_t *) shp_mem_18302)[write_i_17872];\n        int32_t x_17587 = ((__global int32_t *) mem_18324)[write_i_17872];\n        int32_t res_17588 = 1 + write_i_17872;\n        bool cond_17589 = x_17586 == 0;\n        int32_t res_17590;\n        \n        if (cond_17589) {\n            res_17590 = -1;\n        } else {\n            res_17590 = x_17587;\n        }\n        if (sle32(0, res_17590) && slt32(res_17590, aoa_len_17582)) {\n            ((__global int32_t *) mem_18330)[res_17590] = res_17588;\n        }\n    }\n}\n__kernel void segmap_17920(int32_t n_17467, __global\n                           unsigned char *arr_mem_18303, __global\n                           unsigned char *mem_18320, __global\n                           unsigned char *mem_18327, __global\n                           unsigned char *mem_18333, __global\n                           unsigned char *mem_18335)\n{\n    const int32_t segmap_group_sizze_17938 = mainzisegmap_group_sizze_17923;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_18707;\n    int32_t local_tid_18708;\n    int32_t group_sizze_18711;\n    int32_t wave_sizze_18710;\n    int32_t group_tid_18709;\n    \n    global_tid_18707 = get_global_id(0);\n    local_tid_18708 = get_local_id(0);\n    group_sizze_18711 = get_local_size(0);\n    wave_sizz",
                   "e_18710 = LOCKSTEP_WIDTH;\n    group_tid_18709 = get_group_id(0);\n    \n    int32_t phys_tid_17920 = global_tid_18707;\n    int32_t gtid_17919 = group_tid_18709 * segmap_group_sizze_17938 +\n            local_tid_18708;\n    \n    if (slt32(gtid_17919, n_17467)) {\n        float x_17947 = ((__global float *) arr_mem_18303)[gtid_17919];\n        int32_t x_17948 = ((__global int32_t *) mem_18320)[gtid_17919];\n        float x_17949 = ((__global float *) mem_18327)[x_17948];\n        bool res_17950 = x_17947 < x_17949;\n        int32_t res_17951;\n        \n        if (res_17950) {\n            res_17951 = 1;\n        } else {\n            res_17951 = 0;\n        }\n        ((__global int32_t *) mem_18333)[gtid_17919] = res_17951;\n        ((__global bool *) mem_18335)[gtid_17919] = res_17950;\n    }\n}\n__kernel void segmap_18005(unsigned char cond_17483, int32_t sizze_17490,\n                           __global unsigned char *mem_18348, __global\n                           unsigned char *mem_18358, __global\n                           unsigned char *mem_18361)\n{\n    const int32_t segmap_group_sizze_18023 = mainzisegmap_group_sizze_18008;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_18929;\n    int32_t local_tid_18930;\n    int32_t group_sizze_18933;\n    int32_t wave_sizze_18932;\n    int32_t group_tid_18931;\n    \n    global_tid_18929 = get_global_id(0);\n    local_tid_18930 = get_local_id(0);\n    group_sizze_18933 = get_local_size(0);\n    wave_sizze_18932 = LOCKSTEP_WIDTH;\n    group_tid_18931 = get_group_id(0);\n    \n    int32_t phys_tid_18005 = global_tid_18929;\n    int32_t gtid_18004 = group_tid_18931 * segmap_group_sizze_18023 +\n            local_tid_18930;\n    \n    if (slt32(gtid_18004, sizze_17490)) {\n        int32_t res_18033;\n        \n        if (cond_17483) {\n            int32_t x_18031 = ((__global int32_t *) mem_18358)[gtid_18004];\n            int32_t i_18034 = x_18031 - 1;\n            int32_t res_18035 = ((__global int",
                   "32_t *) mem_18348)[i_18034];\n            \n            res_18033 = res_18035;\n        } else {\n            res_18033 = -1;\n        }\n        ((__global int32_t *) mem_18361)[gtid_18004] = res_18033;\n    }\n}\n__kernel void segmap_18046(int32_t sizze_17490, int32_t aoa_len_17711, __global\n                           unsigned char *shp_mem_18302, __global\n                           unsigned char *mem_18371, __global\n                           unsigned char *mem_18374)\n{\n    const int32_t segmap_group_sizze_18050 = mainzisegmap_group_sizze_18049;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_19008;\n    int32_t local_tid_19009;\n    int32_t group_sizze_19012;\n    int32_t wave_sizze_19011;\n    int32_t group_tid_19010;\n    \n    global_tid_19008 = get_global_id(0);\n    local_tid_19009 = get_local_id(0);\n    group_sizze_19012 = get_local_size(0);\n    wave_sizze_19011 = LOCKSTEP_WIDTH;\n    group_tid_19010 = get_group_id(0);\n    \n    int32_t phys_tid_18046 = global_tid_19008;\n    int32_t write_i_18045 = group_tid_19010 * segmap_group_sizze_18050 +\n            local_tid_19009;\n    \n    if (slt32(write_i_18045, sizze_17490)) {\n        int32_t x_17714 = ((__global int32_t *) shp_mem_18302)[write_i_18045];\n        int32_t x_17715 = ((__global int32_t *) mem_18371)[write_i_18045];\n        int32_t index_primexp_18258 = 1 + write_i_18045;\n        bool cond_17717 = x_17714 == 0;\n        int32_t res_17718;\n        \n        if (cond_17717) {\n            res_17718 = -1;\n        } else {\n            res_17718 = x_17715;\n        }\n        if (sle32(0, res_17718) && slt32(res_17718, aoa_len_17711)) {\n            ((__global int32_t *) mem_18374)[res_17718] = index_primexp_18258;\n        }\n    }\n}\n__kernel void segmap_18107(int32_t aoa_len_17711, __global\n                           unsigned char *mem_18368, __global\n                           unsigned char *mem_18381, __global\n                           unsigned char *mem_18384)\n{",
                   "\n    const int32_t segmap_group_sizze_18128 = mainzisegmap_group_sizze_18110;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_19093;\n    int32_t local_tid_19094;\n    int32_t group_sizze_19097;\n    int32_t wave_sizze_19096;\n    int32_t group_tid_19095;\n    \n    global_tid_19093 = get_global_id(0);\n    local_tid_19094 = get_local_id(0);\n    group_sizze_19097 = get_local_size(0);\n    wave_sizze_19096 = LOCKSTEP_WIDTH;\n    group_tid_19095 = get_group_id(0);\n    \n    int32_t phys_tid_18107 = global_tid_19093;\n    int32_t gtid_18106 = group_tid_19095 * segmap_group_sizze_18128 +\n            local_tid_19094;\n    \n    if (slt32(gtid_18106, aoa_len_17711)) {\n        int32_t x_18137 = ((__global int32_t *) mem_18381)[gtid_18106];\n        int32_t res_18142 = x_18137 - 1;\n        int32_t res_18143 = ((__global int32_t *) mem_18368)[res_18142];\n        \n        ((__global int32_t *) mem_18384)[gtid_18106] = res_18143;\n    }\n}\n__kernel void segmap_18145(int32_t n_17467, __global\n                           unsigned char *arr_mem_18303, __global\n                           unsigned char *mem_18335, __global\n                           unsigned char *mem_18342, __global\n                           unsigned char *mem_18348, __global\n                           unsigned char *mem_18354, __global\n                           unsigned char *mem_18361, __global\n                           unsigned char *mem_18364, __global\n                           unsigned char *mem_18384)\n{\n    const int32_t segmap_group_sizze_18149 = mainzisegmap_group_sizze_18148;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_19098;\n    int32_t local_tid_19099;\n    int32_t group_sizze_19102;\n    int32_t wave_sizze_19101;\n    int32_t group_tid_19100;\n    \n    global_tid_19098 = get_global_id(0);\n    local_tid_19099 = get_local_id(0);\n    group_sizze_19102 = get_local_size(0);\n    wave_sizze_191",
                   "01 = LOCKSTEP_WIDTH;\n    group_tid_19100 = get_group_id(0);\n    \n    int32_t phys_tid_18145 = global_tid_19098;\n    int32_t write_i_18144 = group_tid_19100 * segmap_group_sizze_18149 +\n            local_tid_19099;\n    \n    if (slt32(write_i_18144, n_17467)) {\n        int32_t x_17765 = ((__global int32_t *) mem_18354)[write_i_18144];\n        int32_t x_17766 = ((__global int32_t *) mem_18342)[write_i_18144];\n        bool x_17767 = ((__global bool *) mem_18335)[write_i_18144];\n        int32_t x_17769 = ((__global int32_t *) mem_18384)[write_i_18144];\n        float write_value_17770 = ((__global\n                                    float *) arr_mem_18303)[write_i_18144];\n        int32_t y_17771 = ((__global int32_t *) mem_18361)[x_17766];\n        int32_t res_17772 = x_17765 + y_17771;\n        int32_t res_17773;\n        \n        if (x_17767) {\n            int32_t x_17768 = ((__global int32_t *) mem_18348)[write_i_18144];\n            int32_t res_17774 = x_17768 - 1;\n            \n            res_17773 = res_17774;\n        } else {\n            int32_t res_17775 = res_17772 - 1;\n            \n            res_17773 = res_17775;\n        }\n        \n        int32_t res_17776 = x_17769 + res_17773;\n        \n        if (sle32(0, res_17776) && slt32(res_17776, n_17467)) {\n            ((__global float *) mem_18364)[res_17776] = write_value_17770;\n        }\n    }\n}\n__kernel void segmap_18189(int32_t sizze_17490, int32_t num_groups_18207,\n                           __global unsigned char *shp_mem_18302, __global\n                           unsigned char *mem_18395)\n{\n    const int32_t segmap_group_sizze_18206 = mainzisegmap_group_sizze_18192;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_19104;\n    int32_t local_tid_19105;\n    int32_t group_sizze_19108;\n    int32_t wave_sizze_19107;\n    int32_t group_tid_19106;\n    \n    global_tid_19104 = get_global_id(0);\n    local_tid_19105 = get_local_id(0);\n    group_sizze_19108 = get",
                   "_local_size(0);\n    wave_sizze_19107 = LOCKSTEP_WIDTH;\n    group_tid_19106 = get_group_id(0);\n    \n    int32_t phys_tid_18189 = global_tid_19104;\n    int32_t phys_group_id_19109;\n    \n    phys_group_id_19109 = get_group_id(0);\n    for (int32_t i_19110 = 0; i_19110 < squot32(squot32(sizze_17490 +\n                                                        segmap_group_sizze_18206 -\n                                                        1,\n                                                        segmap_group_sizze_18206) -\n                                                phys_group_id_19109 +\n                                                num_groups_18207 - 1,\n                                                num_groups_18207); i_19110++) {\n        int32_t virt_group_id_19111 = phys_group_id_19109 + i_19110 *\n                num_groups_18207;\n        int32_t gtid_18188 = virt_group_id_19111 * segmap_group_sizze_18206 +\n                local_tid_19105;\n        \n        if (slt32(gtid_18188, sizze_17490)) {\n            int32_t x_18210 = ((__global int32_t *) shp_mem_18302)[gtid_18188];\n            bool cond_18212 = x_18210 == 0;\n            __private char *mem_18387;\n            __private char mem_18387_backing_0[8];\n            \n            mem_18387 = mem_18387_backing_0;\n            \n            __private char *mem_18390;\n            __private char mem_18390_backing_1[8];\n            \n            mem_18390 = mem_18390_backing_1;\n            \n            __private char *mem_18420;\n            __private char mem_18420_backing_2[8];\n            \n            mem_18420 = mem_18420_backing_2;\n            if (cond_18212) {\n                for (int32_t i_19112 = 0; i_19112 < 2; i_19112++) {\n                    ((__private int32_t *) mem_18387)[i_19112] = 0;\n                }\n                for (int32_t i_19113 = 0; i_19113 < 2; i_19113++) {\n                    ((__private int32_t *) mem_18420)[i_19113] = ((__private\n                                                                ",
                   "   int32_t *) mem_18387)[i_19113];\n                }\n            } else {\n                int32_t arr_elem_18215 = x_18210 - x_18210;\n                \n                ((__private int32_t *) mem_18390)[0] = x_18210;\n                ((__private int32_t *) mem_18390)[1] = arr_elem_18215;\n                for (int32_t i_19114 = 0; i_19114 < 2; i_19114++) {\n                    ((__private int32_t *) mem_18420)[i_19114] = ((__private\n                                                                   int32_t *) mem_18390)[i_19114];\n                }\n            }\n            for (int32_t i_19115 = 0; i_19115 < 2; i_19115++) {\n                ((__global int32_t *) mem_18395)[gtid_18188 + i_19115 *\n                                                 sizze_17490] = ((__private\n                                                                  int32_t *) mem_18420)[i_19115];\n            }\n        }\n    }\n}\n__kernel void segmap_18227(int32_t partition_sizze_17800,\n                           int32_t convop_x_18393, __global\n                           unsigned char *mem_18399, __global\n                           unsigned char *mem_18403, __global\n                           unsigned char *mem_18406, __global\n                           unsigned char *mem_18409)\n{\n    const int32_t segmap_group_sizze_18231 = mainzisegmap_group_sizze_18230;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t global_tid_19162;\n    int32_t local_tid_19163;\n    int32_t group_sizze_19166;\n    int32_t wave_sizze_19165;\n    int32_t group_tid_19164;\n    \n    global_tid_19162 = get_global_id(0);\n    local_tid_19163 = get_local_id(0);\n    group_sizze_19166 = get_local_size(0);\n    wave_sizze_19165 = LOCKSTEP_WIDTH;\n    group_tid_19164 = get_group_id(0);\n    \n    int32_t phys_tid_18227 = global_tid_19162;\n    int32_t write_i_18226 = group_tid_19164 * segmap_group_sizze_18231 +\n            local_tid_19163;\n    \n    if (slt32(write_i_18226, convop_x_18393)) {\n        ",
                   "int32_t c_17804 = ((__global int32_t *) mem_18406)[write_i_18226];\n        int32_t offset_17805 = ((__global int32_t *) mem_18403)[write_i_18226];\n        int32_t new_index_18263 = squot32(write_i_18226, 2);\n        int32_t binop_y_18265 = 2 * new_index_18263;\n        int32_t new_index_18266 = write_i_18226 - binop_y_18265;\n        int32_t v_17806 = ((__global int32_t *) mem_18399)[new_index_18263 * 2 +\n                                                           new_index_18266];\n        bool is_this_one_17807 = c_17804 == 0;\n        int32_t this_offset_17808 = -1 + offset_17805;\n        int32_t total_res_17809;\n        \n        if (is_this_one_17807) {\n            total_res_17809 = this_offset_17808;\n        } else {\n            total_res_17809 = -1;\n        }\n        if (sle32(0, total_res_17809) && slt32(total_res_17809,\n                                               partition_sizze_17800)) {\n            ((__global int32_t *) mem_18409)[total_res_17809] = v_17806;\n        }\n    }\n}\n__kernel void segred_nonseg_17832(__local volatile\n                                  int64_t *sync_arr_mem_18486_backing_aligned_0,\n                                  __local volatile\n                                  int64_t *red_arr_mem_18488_backing_aligned_1,\n                                  int32_t iota_arg_17470,\n                                  int32_t num_groups_17827, __global\n                                  unsigned char *arr_mem_18295, __global\n                                  unsigned char *mem_18301, __global\n                                  unsigned char *counter_mem_18476, __global\n                                  unsigned char *group_res_arr_mem_18478,\n                                  int32_t num_threads_18480)\n{\n    const int32_t segred_group_sizze_17825 = mainzisegred_group_sizze_17824;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict sync_arr_mem_18486_backing_0 =\n                     ",
                   "     (__local volatile\n                           char *) sync_arr_mem_18486_backing_aligned_0;\n    __local volatile char *restrict red_arr_mem_18488_backing_1 =\n                          (__local volatile\n                           char *) red_arr_mem_18488_backing_aligned_1;\n    int32_t global_tid_18481;\n    int32_t local_tid_18482;\n    int32_t group_sizze_18485;\n    int32_t wave_sizze_18484;\n    int32_t group_tid_18483;\n    \n    global_tid_18481 = get_global_id(0);\n    local_tid_18482 = get_local_id(0);\n    group_sizze_18485 = get_local_size(0);\n    wave_sizze_18484 = LOCKSTEP_WIDTH;\n    group_tid_18483 = get_group_id(0);\n    \n    int32_t phys_tid_17832 = global_tid_18481;\n    __local char *sync_arr_mem_18486;\n    \n    sync_arr_mem_18486 = (__local char *) sync_arr_mem_18486_backing_0;\n    \n    __local char *red_arr_mem_18488;\n    \n    red_arr_mem_18488 = (__local char *) red_arr_mem_18488_backing_1;\n    \n    int32_t dummy_17830 = 0;\n    int32_t gtid_17831;\n    \n    gtid_17831 = 0;\n    \n    bool x_acc_18490;\n    int32_t chunk_sizze_18491 = smin32(squot32(iota_arg_17470 +\n                                               segred_group_sizze_17825 *\n                                               num_groups_17827 - 1,\n                                               segred_group_sizze_17825 *\n                                               num_groups_17827),\n                                       squot32(iota_arg_17470 - phys_tid_17832 +\n                                               num_threads_18480 - 1,\n                                               num_threads_18480));\n    bool x_17474;\n    bool x_17475;\n    \n    // neutral-initialise the accumulators\n    {\n        x_acc_18490 = 1;\n    }\n    for (int32_t i_18495 = 0; i_18495 < chunk_sizze_18491; i_18495++) {\n        gtid_17831 = phys_tid_17832 + num_threads_18480 * i_18495;\n        // apply map function\n        {\n            float arr_elem_17478 = ((__global\n                                     float *) arr_mem_18295)[",
                   "gtid_17831];\n            int32_t i_17479 = 1 + gtid_17831;\n            float y_17480 = ((__global float *) arr_mem_18295)[i_17479];\n            bool res_17481 = arr_elem_17478 <= y_17480;\n            \n            // save map-out results\n            { }\n            // load accumulator\n            {\n                x_17474 = x_acc_18490;\n            }\n            // load new values\n            {\n                x_17475 = res_17481;\n            }\n            // apply reduction operator\n            {\n                bool x_17476 = x_17474 && x_17475;\n                \n                // store in accumulator\n                {\n                    x_acc_18490 = x_17476;\n                }\n            }\n        }\n    }\n    // to reduce current chunk, first store our result in memory\n    {\n        x_17474 = x_acc_18490;\n        ((__local bool *) red_arr_mem_18488)[local_tid_18482] = x_17474;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t offset_18496;\n    int32_t skip_waves_18497;\n    bool x_18492;\n    bool x_18493;\n    \n    offset_18496 = 0;\n    // participating threads read initial accumulator\n    {\n        if (slt32(local_tid_18482, segred_group_sizze_17825)) {\n            x_18492 = ((__local bool *) red_arr_mem_18488)[local_tid_18482 +\n                                                           offset_18496];\n        }\n    }\n    offset_18496 = 1;\n    while (slt32(offset_18496, wave_sizze_18484)) {\n        if (slt32(local_tid_18482 + offset_18496, segred_group_sizze_17825) &&\n            ((local_tid_18482 - squot32(local_tid_18482, wave_sizze_18484) *\n              wave_sizze_18484) & (2 * offset_18496 - 1)) == 0) {\n            // read array element\n            {\n                x_18493 = ((volatile __local\n                            bool *) red_arr_mem_18488)[local_tid_18482 +\n                                                       offset_18496];\n            }\n            // apply reduction operation\n            {\n                bool x_18494 = x_18492 && x_18493;",
                   "\n                \n                x_18492 = x_18494;\n            }\n            // write result of operation\n            {\n                ((volatile __local bool *) red_arr_mem_18488)[local_tid_18482] =\n                    x_18492;\n            }\n        }\n        offset_18496 *= 2;\n    }\n    skip_waves_18497 = 1;\n    while (slt32(skip_waves_18497, squot32(segred_group_sizze_17825 +\n                                           wave_sizze_18484 - 1,\n                                           wave_sizze_18484))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        offset_18496 = skip_waves_18497 * wave_sizze_18484;\n        if (slt32(local_tid_18482 + offset_18496, segred_group_sizze_17825) &&\n            ((local_tid_18482 - squot32(local_tid_18482, wave_sizze_18484) *\n              wave_sizze_18484) == 0 && (squot32(local_tid_18482,\n                                                 wave_sizze_18484) & (2 *\n                                                                      skip_waves_18497 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_18493 = ((__local bool *) red_arr_mem_18488)[local_tid_18482 +\n                                                               offset_18496];\n            }\n            // apply reduction operation\n            {\n                bool x_18494 = x_18492 && x_18493;\n                \n                x_18492 = x_18494;\n            }\n            // write result of operation\n            {\n                ((__local bool *) red_arr_mem_18488)[local_tid_18482] = x_18492;\n            }\n        }\n        skip_waves_18497 *= 2;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // first thread saves the result in accumulator\n    {\n        if (local_tid_18482 == 0) {\n            x_acc_18490 = x_18492;\n        }\n    }\n    \n    int32_t old_counter_18498;\n    \n    // first thread in group saves group result to global memory\n    {\n        if (local_tid_184",
                   "82 == 0) {\n            ((__global bool *) group_res_arr_mem_18478)[group_tid_18483 *\n                                                        segred_group_sizze_17825] =\n                x_acc_18490;\n            mem_fence_global();\n            old_counter_18498 = atomic_add(&((volatile __global\n                                              int *) counter_mem_18476)[0],\n                                           (int) 1);\n            ((__local bool *) sync_arr_mem_18486)[0] = old_counter_18498 ==\n                num_groups_17827 - 1;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    barrier(CLK_GLOBAL_MEM_FENCE);\n    \n    bool is_last_group_18499 = ((__local bool *) sync_arr_mem_18486)[0];\n    \n    if (is_last_group_18499) {\n        if (local_tid_18482 == 0) {\n            old_counter_18498 = atomic_add(&((volatile __global\n                                              int *) counter_mem_18476)[0],\n                                           (int) (0 - num_groups_17827));\n        }\n        // read in the per-group-results\n        {\n            if (slt32(local_tid_18482, num_groups_17827)) {\n                x_17474 = ((__global\n                            bool *) group_res_arr_mem_18478)[local_tid_18482 *\n                                                             segred_group_sizze_17825];\n            } else {\n                x_17474 = 1;\n            }\n            ((__local bool *) red_arr_mem_18488)[local_tid_18482] = x_17474;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // reduce the per-group results\n        {\n            int32_t offset_18500;\n            int32_t skip_waves_18501;\n            bool x_18492;\n            bool x_18493;\n            \n            offset_18500 = 0;\n            // participating threads read initial accumulator\n            {\n                if (slt32(local_tid_18482, segred_group_sizze_17825)) {\n                    x_18492 = ((__local\n                                bool *) red_arr_mem_18488)[local_tid_18482 +\n                  ",
                   "                                         offset_18500];\n                }\n            }\n            offset_18500 = 1;\n            while (slt32(offset_18500, wave_sizze_18484)) {\n                if (slt32(local_tid_18482 + offset_18500,\n                          segred_group_sizze_17825) && ((local_tid_18482 -\n                                                         squot32(local_tid_18482,\n                                                                 wave_sizze_18484) *\n                                                         wave_sizze_18484) &\n                                                        (2 * offset_18500 -\n                                                         1)) == 0) {\n                    // read array element\n                    {\n                        x_18493 = ((volatile __local\n                                    bool *) red_arr_mem_18488)[local_tid_18482 +\n                                                               offset_18500];\n                    }\n                    // apply reduction operation\n                    {\n                        bool x_18494 = x_18492 && x_18493;\n                        \n                        x_18492 = x_18494;\n                    }\n                    // write result of operation\n                    {\n                        ((volatile __local\n                          bool *) red_arr_mem_18488)[local_tid_18482] = x_18492;\n                    }\n                }\n                offset_18500 *= 2;\n            }\n            skip_waves_18501 = 1;\n            while (slt32(skip_waves_18501, squot32(segred_group_sizze_17825 +\n                                                   wave_sizze_18484 - 1,\n                                                   wave_sizze_18484))) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n                offset_18500 = skip_waves_18501 * wave_sizze_18484;\n                if (slt32(local_tid_18482 + offset_18500,\n                          segred_group_sizze_17825) && ((local_tid_1",
                   "8482 -\n                                                         squot32(local_tid_18482,\n                                                                 wave_sizze_18484) *\n                                                         wave_sizze_18484) ==\n                                                        0 &&\n                                                        (squot32(local_tid_18482,\n                                                                 wave_sizze_18484) &\n                                                         (2 * skip_waves_18501 -\n                                                          1)) == 0)) {\n                    // read array element\n                    {\n                        x_18493 = ((__local\n                                    bool *) red_arr_mem_18488)[local_tid_18482 +\n                                                               offset_18500];\n                    }\n                    // apply reduction operation\n                    {\n                        bool x_18494 = x_18492 && x_18493;\n                        \n                        x_18492 = x_18494;\n                    }\n                    // write result of operation\n                    {\n                        ((__local bool *) red_arr_mem_18488)[local_tid_18482] =\n                            x_18492;\n                    }\n                }\n                skip_waves_18501 *= 2;\n            }\n            // and back to memory with the final result\n            {\n                if (local_tid_18482 == 0) {\n                    ((__global bool *) mem_18301)[0] = x_18492;\n                }\n            }\n        }\n    }\n}\n__kernel void segred_nonseg_18248(__local volatile\n                                  int64_t *sync_arr_mem_19177_backing_aligned_0,\n                                  __local volatile\n                                  int64_t *red_arr_mem_19179_backing_aligned_1,\n                                  int32_t iota_arg_17470,\n                             ",
                   "     int32_t num_groups_18243, __global\n                                  unsigned char *mem_18364, __global\n                                  unsigned char *mem_18412, __global\n                                  unsigned char *counter_mem_19167, __global\n                                  unsigned char *group_res_arr_mem_19169,\n                                  int32_t num_threads_19171)\n{\n    const int32_t segred_group_sizze_18241 = mainzisegred_group_sizze_18240;\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    __local volatile char *restrict sync_arr_mem_19177_backing_0 =\n                          (__local volatile\n                           char *) sync_arr_mem_19177_backing_aligned_0;\n    __local volatile char *restrict red_arr_mem_19179_backing_1 =\n                          (__local volatile\n                           char *) red_arr_mem_19179_backing_aligned_1;\n    int32_t global_tid_19172;\n    int32_t local_tid_19173;\n    int32_t group_sizze_19176;\n    int32_t wave_sizze_19175;\n    int32_t group_tid_19174;\n    \n    global_tid_19172 = get_global_id(0);\n    local_tid_19173 = get_local_id(0);\n    group_sizze_19176 = get_local_size(0);\n    wave_sizze_19175 = LOCKSTEP_WIDTH;\n    group_tid_19174 = get_group_id(0);\n    \n    int32_t phys_tid_18248 = global_tid_19172;\n    __local char *sync_arr_mem_19177;\n    \n    sync_arr_mem_19177 = (__local char *) sync_arr_mem_19177_backing_0;\n    \n    __local char *red_arr_mem_19179;\n    \n    red_arr_mem_19179 = (__local char *) red_arr_mem_19179_backing_1;\n    \n    int32_t dummy_18246 = 0;\n    int32_t gtid_18247;\n    \n    gtid_18247 = 0;\n    \n    bool x_acc_19181;\n    int32_t chunk_sizze_19182 = smin32(squot32(iota_arg_17470 +\n                                               segred_group_sizze_18241 *\n                                               num_groups_18243 - 1,\n                                               segred_group_sizze_18241 *\n                                             ",
                   "  num_groups_18243),\n                                       squot32(iota_arg_17470 - phys_tid_18248 +\n                                               num_threads_19171 - 1,\n                                               num_threads_19171));\n    bool x_17812;\n    bool x_17813;\n    \n    // neutral-initialise the accumulators\n    {\n        x_acc_19181 = 1;\n    }\n    for (int32_t i_19186 = 0; i_19186 < chunk_sizze_19182; i_19186++) {\n        gtid_18247 = phys_tid_18248 + num_threads_19171 * i_19186;\n        // apply map function\n        {\n            float res_elem_17816 = ((__global float *) mem_18364)[gtid_18247];\n            int32_t i_17817 = 1 + gtid_18247;\n            float y_17818 = ((__global float *) mem_18364)[i_17817];\n            bool res_17819 = res_elem_17816 <= y_17818;\n            \n            // save map-out results\n            { }\n            // load accumulator\n            {\n                x_17812 = x_acc_19181;\n            }\n            // load new values\n            {\n                x_17813 = res_17819;\n            }\n            // apply reduction operator\n            {\n                bool x_17814 = x_17812 && x_17813;\n                \n                // store in accumulator\n                {\n                    x_acc_19181 = x_17814;\n                }\n            }\n        }\n    }\n    // to reduce current chunk, first store our result in memory\n    {\n        x_17812 = x_acc_19181;\n        ((__local bool *) red_arr_mem_19179)[local_tid_19173] = x_17812;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t offset_19187;\n    int32_t skip_waves_19188;\n    bool x_19183;\n    bool x_19184;\n    \n    offset_19187 = 0;\n    // participating threads read initial accumulator\n    {\n        if (slt32(local_tid_19173, segred_group_sizze_18241)) {\n            x_19183 = ((__local bool *) red_arr_mem_19179)[local_tid_19173 +\n                                                           offset_19187];\n        }\n    }\n    offset_19187 = 1;\n    while (slt32(offset_191",
                   "87, wave_sizze_19175)) {\n        if (slt32(local_tid_19173 + offset_19187, segred_group_sizze_18241) &&\n            ((local_tid_19173 - squot32(local_tid_19173, wave_sizze_19175) *\n              wave_sizze_19175) & (2 * offset_19187 - 1)) == 0) {\n            // read array element\n            {\n                x_19184 = ((volatile __local\n                            bool *) red_arr_mem_19179)[local_tid_19173 +\n                                                       offset_19187];\n            }\n            // apply reduction operation\n            {\n                bool x_19185 = x_19183 && x_19184;\n                \n                x_19183 = x_19185;\n            }\n            // write result of operation\n            {\n                ((volatile __local bool *) red_arr_mem_19179)[local_tid_19173] =\n                    x_19183;\n            }\n        }\n        offset_19187 *= 2;\n    }\n    skip_waves_19188 = 1;\n    while (slt32(skip_waves_19188, squot32(segred_group_sizze_18241 +\n                                           wave_sizze_19175 - 1,\n                                           wave_sizze_19175))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        offset_19187 = skip_waves_19188 * wave_sizze_19175;\n        if (slt32(local_tid_19173 + offset_19187, segred_group_sizze_18241) &&\n            ((local_tid_19173 - squot32(local_tid_19173, wave_sizze_19175) *\n              wave_sizze_19175) == 0 && (squot32(local_tid_19173,\n                                                 wave_sizze_19175) & (2 *\n                                                                      skip_waves_19188 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_19184 = ((__local bool *) red_arr_mem_19179)[local_tid_19173 +\n                                                               offset_19187];\n            }\n            // apply reduction operation\n            {\n                bool x_19185",
                   " = x_19183 && x_19184;\n                \n                x_19183 = x_19185;\n            }\n            // write result of operation\n            {\n                ((__local bool *) red_arr_mem_19179)[local_tid_19173] = x_19183;\n            }\n        }\n        skip_waves_19188 *= 2;\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // first thread saves the result in accumulator\n    {\n        if (local_tid_19173 == 0) {\n            x_acc_19181 = x_19183;\n        }\n    }\n    \n    int32_t old_counter_19189;\n    \n    // first thread in group saves group result to global memory\n    {\n        if (local_tid_19173 == 0) {\n            ((__global bool *) group_res_arr_mem_19169)[group_tid_19174 *\n                                                        segred_group_sizze_18241] =\n                x_acc_19181;\n            mem_fence_global();\n            old_counter_19189 = atomic_add(&((volatile __global\n                                              int *) counter_mem_19167)[0],\n                                           (int) 1);\n            ((__local bool *) sync_arr_mem_19177)[0] = old_counter_19189 ==\n                num_groups_18243 - 1;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    barrier(CLK_GLOBAL_MEM_FENCE);\n    \n    bool is_last_group_19190 = ((__local bool *) sync_arr_mem_19177)[0];\n    \n    if (is_last_group_19190) {\n        if (local_tid_19173 == 0) {\n            old_counter_19189 = atomic_add(&((volatile __global\n                                              int *) counter_mem_19167)[0],\n                                           (int) (0 - num_groups_18243));\n        }\n        // read in the per-group-results\n        {\n            if (slt32(local_tid_19173, num_groups_18243)) {\n                x_17812 = ((__global\n                            bool *) group_res_arr_mem_19169)[local_tid_19173 *\n                                                             segred_group_sizze_18241];\n            } else {\n                x_17812 = 1;\n            }\n            ((__local b",
                   "ool *) red_arr_mem_19179)[local_tid_19173] = x_17812;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // reduce the per-group results\n        {\n            int32_t offset_19191;\n            int32_t skip_waves_19192;\n            bool x_19183;\n            bool x_19184;\n            \n            offset_19191 = 0;\n            // participating threads read initial accumulator\n            {\n                if (slt32(local_tid_19173, segred_group_sizze_18241)) {\n                    x_19183 = ((__local\n                                bool *) red_arr_mem_19179)[local_tid_19173 +\n                                                           offset_19191];\n                }\n            }\n            offset_19191 = 1;\n            while (slt32(offset_19191, wave_sizze_19175)) {\n                if (slt32(local_tid_19173 + offset_19191,\n                          segred_group_sizze_18241) && ((local_tid_19173 -\n                                                         squot32(local_tid_19173,\n                                                                 wave_sizze_19175) *\n                                                         wave_sizze_19175) &\n                                                        (2 * offset_19191 -\n                                                         1)) == 0) {\n                    // read array element\n                    {\n                        x_19184 = ((volatile __local\n                                    bool *) red_arr_mem_19179)[local_tid_19173 +\n                                                               offset_19191];\n                    }\n                    // apply reduction operation\n                    {\n                        bool x_19185 = x_19183 && x_19184;\n                        \n                        x_19183 = x_19185;\n                    }\n                    // write result of operation\n                    {\n                        ((volatile __local\n                          bool *) red_arr_mem_19179)[local_tid_191",
                   "73] = x_19183;\n                    }\n                }\n                offset_19191 *= 2;\n            }\n            skip_waves_19192 = 1;\n            while (slt32(skip_waves_19192, squot32(segred_group_sizze_18241 +\n                                                   wave_sizze_19175 - 1,\n                                                   wave_sizze_19175))) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n                offset_19191 = skip_waves_19192 * wave_sizze_19175;\n                if (slt32(local_tid_19173 + offset_19191,\n                          segred_group_sizze_18241) && ((local_tid_19173 -\n                                                         squot32(local_tid_19173,\n                                                                 wave_sizze_19175) *\n                                                         wave_sizze_19175) ==\n                                                        0 &&\n                                                        (squot32(local_tid_19173,\n                                                                 wave_sizze_19175) &\n                                                         (2 * skip_waves_19192 -\n                                                          1)) == 0)) {\n                    // read array element\n                    {\n                        x_19184 = ((__local\n                                    bool *) red_arr_mem_19179)[local_tid_19173 +\n                                                               offset_19191];\n                    }\n                    // apply reduction operation\n                    {\n                        bool x_19185 = x_19183 && x_19184;\n                        \n                        x_19183 = x_19185;\n                    }\n                    // write result of operation\n                    {\n                        ((__local bool *) red_arr_mem_19179)[local_tid_19173] =\n                            x_19183;\n                    }\n                }\n                skip_waves_19",
                   "192 *= 2;\n            }\n            // and back to memory with the final result\n            {\n                if (local_tid_19173 == 0) {\n                    ((__global bool *) mem_18412)[0] = x_19183;\n                }\n            }\n        }\n    }\n}\n",
                   NULL};
static int32_t counter_mem_realtype_19196[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int32_t counter_mem_realtype_19378[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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
static const char *size_names[] = {"main.group_size_18473",
                                   "main.group_size_18572",
                                   "main.group_size_18652",
                                   "main.group_size_18698",
                                   "main.group_size_18879",
                                   "main.group_size_18925",
                                   "main.group_size_18941",
                                   "main.group_size_19004",
                                   "main.group_size_19089",
                                   "main.group_size_19158",
                                   "main.segmap_group_size_17846",
                                   "main.segmap_group_size_17876",
                                   "main.segmap_group_size_17923",
                                   "main.segmap_group_size_18008",
                                   "main.segmap_group_size_18049",
                                   "main.segmap_group_size_18110",
                                   "main.segmap_group_size_18148",
                                   "main.segmap_group_size_18192",
                                   "main.segmap_group_size_18230",
                                   "main.segmap_num_groups_18194",
                                   "main.segred_group_size_17824",
                                   "main.segred_group_size_18240",
                                   "main.segred_num_groups_17826",
                                   "main.segred_num_groups_18242",
                                   "main.segscan_group_size_17835",
                                   "main.segscan_group_size_17856",
                                   "main.segscan_group_size_17865",
                                   "main.segscan_group_size_17954",
                                   "main.segscan_group_size_17963",
                                   "main.segscan_group_size_18038",
                                   "main.segscan_group_size_18059",
                                   "main.segscan_group_size_18219",
                                   "main.segscan_num_groups_17837",
                                   "main.segscan_num_groups_17858",
                                   "main.segscan_num_groups_17867",
                                   "main.segscan_num_groups_17956",
                                   "main.segscan_num_groups_17965",
                                   "main.segscan_num_groups_18040",
                                   "main.segscan_num_groups_18061",
                                   "main.segscan_num_groups_18221"};
static const char *size_vars[] = {"mainzigroup_sizze_18473",
                                  "mainzigroup_sizze_18572",
                                  "mainzigroup_sizze_18652",
                                  "mainzigroup_sizze_18698",
                                  "mainzigroup_sizze_18879",
                                  "mainzigroup_sizze_18925",
                                  "mainzigroup_sizze_18941",
                                  "mainzigroup_sizze_19004",
                                  "mainzigroup_sizze_19089",
                                  "mainzigroup_sizze_19158",
                                  "mainzisegmap_group_sizze_17846",
                                  "mainzisegmap_group_sizze_17876",
                                  "mainzisegmap_group_sizze_17923",
                                  "mainzisegmap_group_sizze_18008",
                                  "mainzisegmap_group_sizze_18049",
                                  "mainzisegmap_group_sizze_18110",
                                  "mainzisegmap_group_sizze_18148",
                                  "mainzisegmap_group_sizze_18192",
                                  "mainzisegmap_group_sizze_18230",
                                  "mainzisegmap_num_groups_18194",
                                  "mainzisegred_group_sizze_17824",
                                  "mainzisegred_group_sizze_18240",
                                  "mainzisegred_num_groups_17826",
                                  "mainzisegred_num_groups_18242",
                                  "mainzisegscan_group_sizze_17835",
                                  "mainzisegscan_group_sizze_17856",
                                  "mainzisegscan_group_sizze_17865",
                                  "mainzisegscan_group_sizze_17954",
                                  "mainzisegscan_group_sizze_17963",
                                  "mainzisegscan_group_sizze_18038",
                                  "mainzisegscan_group_sizze_18059",
                                  "mainzisegscan_group_sizze_18219",
                                  "mainzisegscan_num_groups_17837",
                                  "mainzisegscan_num_groups_17858",
                                  "mainzisegscan_num_groups_17867",
                                  "mainzisegscan_num_groups_17956",
                                  "mainzisegscan_num_groups_17965",
                                  "mainzisegscan_num_groups_18040",
                                  "mainzisegscan_num_groups_18061",
                                  "mainzisegscan_num_groups_18221"};
static const char *size_classes[] = {"group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "num_groups", "group_size",
                                     "group_size", "num_groups", "num_groups",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "num_groups",
                                     "num_groups", "num_groups", "num_groups",
                                     "num_groups", "num_groups", "num_groups",
                                     "num_groups"};
int futhark_get_num_sizes(void)
{
    return 40;
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
    size_t mainzigroup_sizze_18473;
    size_t mainzigroup_sizze_18572;
    size_t mainzigroup_sizze_18652;
    size_t mainzigroup_sizze_18698;
    size_t mainzigroup_sizze_18879;
    size_t mainzigroup_sizze_18925;
    size_t mainzigroup_sizze_18941;
    size_t mainzigroup_sizze_19004;
    size_t mainzigroup_sizze_19089;
    size_t mainzigroup_sizze_19158;
    size_t mainzisegmap_group_sizze_17846;
    size_t mainzisegmap_group_sizze_17876;
    size_t mainzisegmap_group_sizze_17923;
    size_t mainzisegmap_group_sizze_18008;
    size_t mainzisegmap_group_sizze_18049;
    size_t mainzisegmap_group_sizze_18110;
    size_t mainzisegmap_group_sizze_18148;
    size_t mainzisegmap_group_sizze_18192;
    size_t mainzisegmap_group_sizze_18230;
    size_t mainzisegmap_num_groups_18194;
    size_t mainzisegred_group_sizze_17824;
    size_t mainzisegred_group_sizze_18240;
    size_t mainzisegred_num_groups_17826;
    size_t mainzisegred_num_groups_18242;
    size_t mainzisegscan_group_sizze_17835;
    size_t mainzisegscan_group_sizze_17856;
    size_t mainzisegscan_group_sizze_17865;
    size_t mainzisegscan_group_sizze_17954;
    size_t mainzisegscan_group_sizze_17963;
    size_t mainzisegscan_group_sizze_18038;
    size_t mainzisegscan_group_sizze_18059;
    size_t mainzisegscan_group_sizze_18219;
    size_t mainzisegscan_num_groups_17837;
    size_t mainzisegscan_num_groups_17858;
    size_t mainzisegscan_num_groups_17867;
    size_t mainzisegscan_num_groups_17956;
    size_t mainzisegscan_num_groups_17965;
    size_t mainzisegscan_num_groups_18040;
    size_t mainzisegscan_num_groups_18061;
    size_t mainzisegscan_num_groups_18221;
} ;
struct futhark_context_config {
    struct opencl_config opencl;
    size_t sizes[40];
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
    cfg->sizes[9] = 0;
    cfg->sizes[10] = 0;
    cfg->sizes[11] = 0;
    cfg->sizes[12] = 0;
    cfg->sizes[13] = 0;
    cfg->sizes[14] = 0;
    cfg->sizes[15] = 0;
    cfg->sizes[16] = 0;
    cfg->sizes[17] = 0;
    cfg->sizes[18] = 0;
    cfg->sizes[19] = 0;
    cfg->sizes[20] = 0;
    cfg->sizes[21] = 0;
    cfg->sizes[22] = 0;
    cfg->sizes[23] = 0;
    cfg->sizes[24] = 0;
    cfg->sizes[25] = 0;
    cfg->sizes[26] = 0;
    cfg->sizes[27] = 0;
    cfg->sizes[28] = 0;
    cfg->sizes[29] = 0;
    cfg->sizes[30] = 0;
    cfg->sizes[31] = 0;
    cfg->sizes[32] = 0;
    cfg->sizes[33] = 0;
    cfg->sizes[34] = 0;
    cfg->sizes[35] = 0;
    cfg->sizes[36] = 0;
    cfg->sizes[37] = 0;
    cfg->sizes[38] = 0;
    cfg->sizes[39] = 0;
    opencl_config_init(&cfg->opencl, 40, size_names, size_vars, cfg->sizes,
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
    for (int i = 0; i < 40; i++) {
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
    struct memblock_device counter_mem_18476;
    struct memblock_device counter_mem_19167;
    int total_runs;
    long total_runtime;
    cl_kernel map_transpose_i32;
    cl_kernel map_transpose_i32_low_height;
    cl_kernel map_transpose_i32_low_width;
    cl_kernel map_transpose_i32_small;
    cl_kernel replicate_18470;
    cl_kernel replicate_18938;
    cl_kernel scan_stage1_17841;
    cl_kernel scan_stage1_17862;
    cl_kernel scan_stage1_17871;
    cl_kernel scan_stage1_17960;
    cl_kernel scan_stage1_17969;
    cl_kernel scan_stage1_18044;
    cl_kernel scan_stage1_18065;
    cl_kernel scan_stage1_18225;
    cl_kernel scan_stage2_17841;
    cl_kernel scan_stage2_17862;
    cl_kernel scan_stage2_17871;
    cl_kernel scan_stage2_17960;
    cl_kernel scan_stage2_17969;
    cl_kernel scan_stage2_18044;
    cl_kernel scan_stage2_18065;
    cl_kernel scan_stage2_18225;
    cl_kernel scan_stage3_18569;
    cl_kernel scan_stage3_18649;
    cl_kernel scan_stage3_18695;
    cl_kernel scan_stage3_18876;
    cl_kernel scan_stage3_18922;
    cl_kernel scan_stage3_19001;
    cl_kernel scan_stage3_19086;
    cl_kernel scan_stage3_19155;
    cl_kernel segmap_17843;
    cl_kernel segmap_17873;
    cl_kernel segmap_17920;
    cl_kernel segmap_18005;
    cl_kernel segmap_18046;
    cl_kernel segmap_18107;
    cl_kernel segmap_18145;
    cl_kernel segmap_18189;
    cl_kernel segmap_18227;
    cl_kernel segred_nonseg_17832;
    cl_kernel segred_nonseg_18248;
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
    int64_t map_transpose_i32_total_runtime;
    int map_transpose_i32_runs;
    int64_t map_transpose_i32_low_height_total_runtime;
    int map_transpose_i32_low_height_runs;
    int64_t map_transpose_i32_low_width_total_runtime;
    int map_transpose_i32_low_width_runs;
    int64_t map_transpose_i32_small_total_runtime;
    int map_transpose_i32_small_runs;
    int64_t replicate_18470_total_runtime;
    int replicate_18470_runs;
    int64_t replicate_18938_total_runtime;
    int replicate_18938_runs;
    int64_t scan_stage1_17841_total_runtime;
    int scan_stage1_17841_runs;
    int64_t scan_stage1_17862_total_runtime;
    int scan_stage1_17862_runs;
    int64_t scan_stage1_17871_total_runtime;
    int scan_stage1_17871_runs;
    int64_t scan_stage1_17960_total_runtime;
    int scan_stage1_17960_runs;
    int64_t scan_stage1_17969_total_runtime;
    int scan_stage1_17969_runs;
    int64_t scan_stage1_18044_total_runtime;
    int scan_stage1_18044_runs;
    int64_t scan_stage1_18065_total_runtime;
    int scan_stage1_18065_runs;
    int64_t scan_stage1_18225_total_runtime;
    int scan_stage1_18225_runs;
    int64_t scan_stage2_17841_total_runtime;
    int scan_stage2_17841_runs;
    int64_t scan_stage2_17862_total_runtime;
    int scan_stage2_17862_runs;
    int64_t scan_stage2_17871_total_runtime;
    int scan_stage2_17871_runs;
    int64_t scan_stage2_17960_total_runtime;
    int scan_stage2_17960_runs;
    int64_t scan_stage2_17969_total_runtime;
    int scan_stage2_17969_runs;
    int64_t scan_stage2_18044_total_runtime;
    int scan_stage2_18044_runs;
    int64_t scan_stage2_18065_total_runtime;
    int scan_stage2_18065_runs;
    int64_t scan_stage2_18225_total_runtime;
    int scan_stage2_18225_runs;
    int64_t scan_stage3_18569_total_runtime;
    int scan_stage3_18569_runs;
    int64_t scan_stage3_18649_total_runtime;
    int scan_stage3_18649_runs;
    int64_t scan_stage3_18695_total_runtime;
    int scan_stage3_18695_runs;
    int64_t scan_stage3_18876_total_runtime;
    int scan_stage3_18876_runs;
    int64_t scan_stage3_18922_total_runtime;
    int scan_stage3_18922_runs;
    int64_t scan_stage3_19001_total_runtime;
    int scan_stage3_19001_runs;
    int64_t scan_stage3_19086_total_runtime;
    int scan_stage3_19086_runs;
    int64_t scan_stage3_19155_total_runtime;
    int scan_stage3_19155_runs;
    int64_t segmap_17843_total_runtime;
    int segmap_17843_runs;
    int64_t segmap_17873_total_runtime;
    int segmap_17873_runs;
    int64_t segmap_17920_total_runtime;
    int segmap_17920_runs;
    int64_t segmap_18005_total_runtime;
    int segmap_18005_runs;
    int64_t segmap_18046_total_runtime;
    int segmap_18046_runs;
    int64_t segmap_18107_total_runtime;
    int segmap_18107_runs;
    int64_t segmap_18145_total_runtime;
    int segmap_18145_runs;
    int64_t segmap_18189_total_runtime;
    int segmap_18189_runs;
    int64_t segmap_18227_total_runtime;
    int segmap_18227_runs;
    int64_t segred_nonseg_17832_total_runtime;
    int segred_nonseg_17832_runs;
    int64_t segred_nonseg_18248_total_runtime;
    int segred_nonseg_18248_runs;
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
    ctx->map_transpose_i32_total_runtime = 0;
    ctx->map_transpose_i32_runs = 0;
    ctx->map_transpose_i32_low_height_total_runtime = 0;
    ctx->map_transpose_i32_low_height_runs = 0;
    ctx->map_transpose_i32_low_width_total_runtime = 0;
    ctx->map_transpose_i32_low_width_runs = 0;
    ctx->map_transpose_i32_small_total_runtime = 0;
    ctx->map_transpose_i32_small_runs = 0;
    ctx->replicate_18470_total_runtime = 0;
    ctx->replicate_18470_runs = 0;
    ctx->replicate_18938_total_runtime = 0;
    ctx->replicate_18938_runs = 0;
    ctx->scan_stage1_17841_total_runtime = 0;
    ctx->scan_stage1_17841_runs = 0;
    ctx->scan_stage1_17862_total_runtime = 0;
    ctx->scan_stage1_17862_runs = 0;
    ctx->scan_stage1_17871_total_runtime = 0;
    ctx->scan_stage1_17871_runs = 0;
    ctx->scan_stage1_17960_total_runtime = 0;
    ctx->scan_stage1_17960_runs = 0;
    ctx->scan_stage1_17969_total_runtime = 0;
    ctx->scan_stage1_17969_runs = 0;
    ctx->scan_stage1_18044_total_runtime = 0;
    ctx->scan_stage1_18044_runs = 0;
    ctx->scan_stage1_18065_total_runtime = 0;
    ctx->scan_stage1_18065_runs = 0;
    ctx->scan_stage1_18225_total_runtime = 0;
    ctx->scan_stage1_18225_runs = 0;
    ctx->scan_stage2_17841_total_runtime = 0;
    ctx->scan_stage2_17841_runs = 0;
    ctx->scan_stage2_17862_total_runtime = 0;
    ctx->scan_stage2_17862_runs = 0;
    ctx->scan_stage2_17871_total_runtime = 0;
    ctx->scan_stage2_17871_runs = 0;
    ctx->scan_stage2_17960_total_runtime = 0;
    ctx->scan_stage2_17960_runs = 0;
    ctx->scan_stage2_17969_total_runtime = 0;
    ctx->scan_stage2_17969_runs = 0;
    ctx->scan_stage2_18044_total_runtime = 0;
    ctx->scan_stage2_18044_runs = 0;
    ctx->scan_stage2_18065_total_runtime = 0;
    ctx->scan_stage2_18065_runs = 0;
    ctx->scan_stage2_18225_total_runtime = 0;
    ctx->scan_stage2_18225_runs = 0;
    ctx->scan_stage3_18569_total_runtime = 0;
    ctx->scan_stage3_18569_runs = 0;
    ctx->scan_stage3_18649_total_runtime = 0;
    ctx->scan_stage3_18649_runs = 0;
    ctx->scan_stage3_18695_total_runtime = 0;
    ctx->scan_stage3_18695_runs = 0;
    ctx->scan_stage3_18876_total_runtime = 0;
    ctx->scan_stage3_18876_runs = 0;
    ctx->scan_stage3_18922_total_runtime = 0;
    ctx->scan_stage3_18922_runs = 0;
    ctx->scan_stage3_19001_total_runtime = 0;
    ctx->scan_stage3_19001_runs = 0;
    ctx->scan_stage3_19086_total_runtime = 0;
    ctx->scan_stage3_19086_runs = 0;
    ctx->scan_stage3_19155_total_runtime = 0;
    ctx->scan_stage3_19155_runs = 0;
    ctx->segmap_17843_total_runtime = 0;
    ctx->segmap_17843_runs = 0;
    ctx->segmap_17873_total_runtime = 0;
    ctx->segmap_17873_runs = 0;
    ctx->segmap_17920_total_runtime = 0;
    ctx->segmap_17920_runs = 0;
    ctx->segmap_18005_total_runtime = 0;
    ctx->segmap_18005_runs = 0;
    ctx->segmap_18046_total_runtime = 0;
    ctx->segmap_18046_runs = 0;
    ctx->segmap_18107_total_runtime = 0;
    ctx->segmap_18107_runs = 0;
    ctx->segmap_18145_total_runtime = 0;
    ctx->segmap_18145_runs = 0;
    ctx->segmap_18189_total_runtime = 0;
    ctx->segmap_18189_runs = 0;
    ctx->segmap_18227_total_runtime = 0;
    ctx->segmap_18227_runs = 0;
    ctx->segred_nonseg_17832_total_runtime = 0;
    ctx->segred_nonseg_17832_runs = 0;
    ctx->segred_nonseg_18248_total_runtime = 0;
    ctx->segred_nonseg_18248_runs = 0;
}
static int init_context_late(struct futhark_context_config *cfg,
                             struct futhark_context *ctx, cl_program prog)
{
    cl_int error;
    
    {
        ctx->map_transpose_i32 = clCreateKernel(prog, "map_transpose_i32",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_transpose_i32");
    }
    {
        ctx->map_transpose_i32_low_height = clCreateKernel(prog,
                                                           "map_transpose_i32_low_height",
                                                           &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "map_transpose_i32_low_height");
    }
    {
        ctx->map_transpose_i32_low_width = clCreateKernel(prog,
                                                          "map_transpose_i32_low_width",
                                                          &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "map_transpose_i32_low_width");
    }
    {
        ctx->map_transpose_i32_small = clCreateKernel(prog,
                                                      "map_transpose_i32_small",
                                                      &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_transpose_i32_small");
    }
    {
        ctx->replicate_18470 = clCreateKernel(prog, "replicate_18470", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "replicate_18470");
    }
    {
        ctx->replicate_18938 = clCreateKernel(prog, "replicate_18938", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "replicate_18938");
    }
    {
        ctx->scan_stage1_17841 = clCreateKernel(prog, "scan_stage1_17841",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage1_17841");
    }
    {
        ctx->scan_stage1_17862 = clCreateKernel(prog, "scan_stage1_17862",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage1_17862");
    }
    {
        ctx->scan_stage1_17871 = clCreateKernel(prog, "scan_stage1_17871",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage1_17871");
    }
    {
        ctx->scan_stage1_17960 = clCreateKernel(prog, "scan_stage1_17960",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage1_17960");
    }
    {
        ctx->scan_stage1_17969 = clCreateKernel(prog, "scan_stage1_17969",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage1_17969");
    }
    {
        ctx->scan_stage1_18044 = clCreateKernel(prog, "scan_stage1_18044",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage1_18044");
    }
    {
        ctx->scan_stage1_18065 = clCreateKernel(prog, "scan_stage1_18065",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage1_18065");
    }
    {
        ctx->scan_stage1_18225 = clCreateKernel(prog, "scan_stage1_18225",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage1_18225");
    }
    {
        ctx->scan_stage2_17841 = clCreateKernel(prog, "scan_stage2_17841",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage2_17841");
    }
    {
        ctx->scan_stage2_17862 = clCreateKernel(prog, "scan_stage2_17862",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage2_17862");
    }
    {
        ctx->scan_stage2_17871 = clCreateKernel(prog, "scan_stage2_17871",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage2_17871");
    }
    {
        ctx->scan_stage2_17960 = clCreateKernel(prog, "scan_stage2_17960",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage2_17960");
    }
    {
        ctx->scan_stage2_17969 = clCreateKernel(prog, "scan_stage2_17969",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage2_17969");
    }
    {
        ctx->scan_stage2_18044 = clCreateKernel(prog, "scan_stage2_18044",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage2_18044");
    }
    {
        ctx->scan_stage2_18065 = clCreateKernel(prog, "scan_stage2_18065",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage2_18065");
    }
    {
        ctx->scan_stage2_18225 = clCreateKernel(prog, "scan_stage2_18225",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage2_18225");
    }
    {
        ctx->scan_stage3_18569 = clCreateKernel(prog, "scan_stage3_18569",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage3_18569");
    }
    {
        ctx->scan_stage3_18649 = clCreateKernel(prog, "scan_stage3_18649",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage3_18649");
    }
    {
        ctx->scan_stage3_18695 = clCreateKernel(prog, "scan_stage3_18695",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage3_18695");
    }
    {
        ctx->scan_stage3_18876 = clCreateKernel(prog, "scan_stage3_18876",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage3_18876");
    }
    {
        ctx->scan_stage3_18922 = clCreateKernel(prog, "scan_stage3_18922",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage3_18922");
    }
    {
        ctx->scan_stage3_19001 = clCreateKernel(prog, "scan_stage3_19001",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage3_19001");
    }
    {
        ctx->scan_stage3_19086 = clCreateKernel(prog, "scan_stage3_19086",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage3_19086");
    }
    {
        ctx->scan_stage3_19155 = clCreateKernel(prog, "scan_stage3_19155",
                                                &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "scan_stage3_19155");
    }
    {
        ctx->segmap_17843 = clCreateKernel(prog, "segmap_17843", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_17843");
    }
    {
        ctx->segmap_17873 = clCreateKernel(prog, "segmap_17873", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_17873");
    }
    {
        ctx->segmap_17920 = clCreateKernel(prog, "segmap_17920", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_17920");
    }
    {
        ctx->segmap_18005 = clCreateKernel(prog, "segmap_18005", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_18005");
    }
    {
        ctx->segmap_18046 = clCreateKernel(prog, "segmap_18046", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_18046");
    }
    {
        ctx->segmap_18107 = clCreateKernel(prog, "segmap_18107", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_18107");
    }
    {
        ctx->segmap_18145 = clCreateKernel(prog, "segmap_18145", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_18145");
    }
    {
        ctx->segmap_18189 = clCreateKernel(prog, "segmap_18189", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_18189");
    }
    {
        ctx->segmap_18227 = clCreateKernel(prog, "segmap_18227", &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segmap_18227");
    }
    {
        ctx->segred_nonseg_17832 = clCreateKernel(prog, "segred_nonseg_17832",
                                                  &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segred_nonseg_17832");
    }
    {
        ctx->segred_nonseg_18248 = clCreateKernel(prog, "segred_nonseg_18248",
                                                  &error);
        OPENCL_SUCCEED_FATAL(error);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "segred_nonseg_18248");
    }
    {
        cl_int success;
        
        ctx->counter_mem_18476.references = NULL;
        ctx->counter_mem_18476.size = 0;
        ctx->counter_mem_18476.mem = clCreateBuffer(ctx->opencl.ctx,
                                                    CL_MEM_READ_WRITE, (10 >
                                                                        0 ? 10 : 1) *
                                                    sizeof(int32_t), NULL,
                                                    &success);
        OPENCL_SUCCEED_OR_RETURN(success);
        if (10 > 0)
            OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                          ctx->counter_mem_18476.mem,
                                                          CL_TRUE, 0, 10 *
                                                          sizeof(int32_t),
                                                          counter_mem_realtype_19196,
                                                          0, NULL, NULL));
    }
    {
        cl_int success;
        
        ctx->counter_mem_19167.references = NULL;
        ctx->counter_mem_19167.size = 0;
        ctx->counter_mem_19167.mem = clCreateBuffer(ctx->opencl.ctx,
                                                    CL_MEM_READ_WRITE, (10 >
                                                                        0 ? 10 : 1) *
                                                    sizeof(int32_t), NULL,
                                                    &success);
        OPENCL_SUCCEED_OR_RETURN(success);
        if (10 > 0)
            OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                          ctx->counter_mem_19167.mem,
                                                          CL_TRUE, 0, 10 *
                                                          sizeof(int32_t),
                                                          counter_mem_realtype_19378,
                                                          0, NULL, NULL));
    }
    ctx->sizes.mainzigroup_sizze_18473 = cfg->sizes[0];
    ctx->sizes.mainzigroup_sizze_18572 = cfg->sizes[1];
    ctx->sizes.mainzigroup_sizze_18652 = cfg->sizes[2];
    ctx->sizes.mainzigroup_sizze_18698 = cfg->sizes[3];
    ctx->sizes.mainzigroup_sizze_18879 = cfg->sizes[4];
    ctx->sizes.mainzigroup_sizze_18925 = cfg->sizes[5];
    ctx->sizes.mainzigroup_sizze_18941 = cfg->sizes[6];
    ctx->sizes.mainzigroup_sizze_19004 = cfg->sizes[7];
    ctx->sizes.mainzigroup_sizze_19089 = cfg->sizes[8];
    ctx->sizes.mainzigroup_sizze_19158 = cfg->sizes[9];
    ctx->sizes.mainzisegmap_group_sizze_17846 = cfg->sizes[10];
    ctx->sizes.mainzisegmap_group_sizze_17876 = cfg->sizes[11];
    ctx->sizes.mainzisegmap_group_sizze_17923 = cfg->sizes[12];
    ctx->sizes.mainzisegmap_group_sizze_18008 = cfg->sizes[13];
    ctx->sizes.mainzisegmap_group_sizze_18049 = cfg->sizes[14];
    ctx->sizes.mainzisegmap_group_sizze_18110 = cfg->sizes[15];
    ctx->sizes.mainzisegmap_group_sizze_18148 = cfg->sizes[16];
    ctx->sizes.mainzisegmap_group_sizze_18192 = cfg->sizes[17];
    ctx->sizes.mainzisegmap_group_sizze_18230 = cfg->sizes[18];
    ctx->sizes.mainzisegmap_num_groups_18194 = cfg->sizes[19];
    ctx->sizes.mainzisegred_group_sizze_17824 = cfg->sizes[20];
    ctx->sizes.mainzisegred_group_sizze_18240 = cfg->sizes[21];
    ctx->sizes.mainzisegred_num_groups_17826 = cfg->sizes[22];
    ctx->sizes.mainzisegred_num_groups_18242 = cfg->sizes[23];
    ctx->sizes.mainzisegscan_group_sizze_17835 = cfg->sizes[24];
    ctx->sizes.mainzisegscan_group_sizze_17856 = cfg->sizes[25];
    ctx->sizes.mainzisegscan_group_sizze_17865 = cfg->sizes[26];
    ctx->sizes.mainzisegscan_group_sizze_17954 = cfg->sizes[27];
    ctx->sizes.mainzisegscan_group_sizze_17963 = cfg->sizes[28];
    ctx->sizes.mainzisegscan_group_sizze_18038 = cfg->sizes[29];
    ctx->sizes.mainzisegscan_group_sizze_18059 = cfg->sizes[30];
    ctx->sizes.mainzisegscan_group_sizze_18219 = cfg->sizes[31];
    ctx->sizes.mainzisegscan_num_groups_17837 = cfg->sizes[32];
    ctx->sizes.mainzisegscan_num_groups_17858 = cfg->sizes[33];
    ctx->sizes.mainzisegscan_num_groups_17867 = cfg->sizes[34];
    ctx->sizes.mainzisegscan_num_groups_17956 = cfg->sizes[35];
    ctx->sizes.mainzisegscan_num_groups_17965 = cfg->sizes[36];
    ctx->sizes.mainzisegscan_num_groups_18040 = cfg->sizes[37];
    ctx->sizes.mainzisegscan_num_groups_18061 = cfg->sizes[38];
    ctx->sizes.mainzisegscan_num_groups_18221 = cfg->sizes[39];
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
                "map_transpose_i32            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->map_transpose_i32_runs,
                (long) ctx->map_transpose_i32_total_runtime /
                (ctx->map_transpose_i32_runs !=
                 0 ? ctx->map_transpose_i32_runs : 1),
                (long) ctx->map_transpose_i32_total_runtime);
        ctx->total_runtime += ctx->map_transpose_i32_total_runtime;
        ctx->total_runs += ctx->map_transpose_i32_runs;
        fprintf(stderr,
                "map_transpose_i32_low_height ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->map_transpose_i32_low_height_runs,
                (long) ctx->map_transpose_i32_low_height_total_runtime /
                (ctx->map_transpose_i32_low_height_runs !=
                 0 ? ctx->map_transpose_i32_low_height_runs : 1),
                (long) ctx->map_transpose_i32_low_height_total_runtime);
        ctx->total_runtime += ctx->map_transpose_i32_low_height_total_runtime;
        ctx->total_runs += ctx->map_transpose_i32_low_height_runs;
        fprintf(stderr,
                "map_transpose_i32_low_width  ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->map_transpose_i32_low_width_runs,
                (long) ctx->map_transpose_i32_low_width_total_runtime /
                (ctx->map_transpose_i32_low_width_runs !=
                 0 ? ctx->map_transpose_i32_low_width_runs : 1),
                (long) ctx->map_transpose_i32_low_width_total_runtime);
        ctx->total_runtime += ctx->map_transpose_i32_low_width_total_runtime;
        ctx->total_runs += ctx->map_transpose_i32_low_width_runs;
        fprintf(stderr,
                "map_transpose_i32_small      ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->map_transpose_i32_small_runs,
                (long) ctx->map_transpose_i32_small_total_runtime /
                (ctx->map_transpose_i32_small_runs !=
                 0 ? ctx->map_transpose_i32_small_runs : 1),
                (long) ctx->map_transpose_i32_small_total_runtime);
        ctx->total_runtime += ctx->map_transpose_i32_small_total_runtime;
        ctx->total_runs += ctx->map_transpose_i32_small_runs;
        fprintf(stderr,
                "replicate_18470              ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->replicate_18470_runs,
                (long) ctx->replicate_18470_total_runtime /
                (ctx->replicate_18470_runs !=
                 0 ? ctx->replicate_18470_runs : 1),
                (long) ctx->replicate_18470_total_runtime);
        ctx->total_runtime += ctx->replicate_18470_total_runtime;
        ctx->total_runs += ctx->replicate_18470_runs;
        fprintf(stderr,
                "replicate_18938              ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->replicate_18938_runs,
                (long) ctx->replicate_18938_total_runtime /
                (ctx->replicate_18938_runs !=
                 0 ? ctx->replicate_18938_runs : 1),
                (long) ctx->replicate_18938_total_runtime);
        ctx->total_runtime += ctx->replicate_18938_total_runtime;
        ctx->total_runs += ctx->replicate_18938_runs;
        fprintf(stderr,
                "scan_stage1_17841            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage1_17841_runs,
                (long) ctx->scan_stage1_17841_total_runtime /
                (ctx->scan_stage1_17841_runs !=
                 0 ? ctx->scan_stage1_17841_runs : 1),
                (long) ctx->scan_stage1_17841_total_runtime);
        ctx->total_runtime += ctx->scan_stage1_17841_total_runtime;
        ctx->total_runs += ctx->scan_stage1_17841_runs;
        fprintf(stderr,
                "scan_stage1_17862            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage1_17862_runs,
                (long) ctx->scan_stage1_17862_total_runtime /
                (ctx->scan_stage1_17862_runs !=
                 0 ? ctx->scan_stage1_17862_runs : 1),
                (long) ctx->scan_stage1_17862_total_runtime);
        ctx->total_runtime += ctx->scan_stage1_17862_total_runtime;
        ctx->total_runs += ctx->scan_stage1_17862_runs;
        fprintf(stderr,
                "scan_stage1_17871            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage1_17871_runs,
                (long) ctx->scan_stage1_17871_total_runtime /
                (ctx->scan_stage1_17871_runs !=
                 0 ? ctx->scan_stage1_17871_runs : 1),
                (long) ctx->scan_stage1_17871_total_runtime);
        ctx->total_runtime += ctx->scan_stage1_17871_total_runtime;
        ctx->total_runs += ctx->scan_stage1_17871_runs;
        fprintf(stderr,
                "scan_stage1_17960            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage1_17960_runs,
                (long) ctx->scan_stage1_17960_total_runtime /
                (ctx->scan_stage1_17960_runs !=
                 0 ? ctx->scan_stage1_17960_runs : 1),
                (long) ctx->scan_stage1_17960_total_runtime);
        ctx->total_runtime += ctx->scan_stage1_17960_total_runtime;
        ctx->total_runs += ctx->scan_stage1_17960_runs;
        fprintf(stderr,
                "scan_stage1_17969            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage1_17969_runs,
                (long) ctx->scan_stage1_17969_total_runtime /
                (ctx->scan_stage1_17969_runs !=
                 0 ? ctx->scan_stage1_17969_runs : 1),
                (long) ctx->scan_stage1_17969_total_runtime);
        ctx->total_runtime += ctx->scan_stage1_17969_total_runtime;
        ctx->total_runs += ctx->scan_stage1_17969_runs;
        fprintf(stderr,
                "scan_stage1_18044            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage1_18044_runs,
                (long) ctx->scan_stage1_18044_total_runtime /
                (ctx->scan_stage1_18044_runs !=
                 0 ? ctx->scan_stage1_18044_runs : 1),
                (long) ctx->scan_stage1_18044_total_runtime);
        ctx->total_runtime += ctx->scan_stage1_18044_total_runtime;
        ctx->total_runs += ctx->scan_stage1_18044_runs;
        fprintf(stderr,
                "scan_stage1_18065            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage1_18065_runs,
                (long) ctx->scan_stage1_18065_total_runtime /
                (ctx->scan_stage1_18065_runs !=
                 0 ? ctx->scan_stage1_18065_runs : 1),
                (long) ctx->scan_stage1_18065_total_runtime);
        ctx->total_runtime += ctx->scan_stage1_18065_total_runtime;
        ctx->total_runs += ctx->scan_stage1_18065_runs;
        fprintf(stderr,
                "scan_stage1_18225            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage1_18225_runs,
                (long) ctx->scan_stage1_18225_total_runtime /
                (ctx->scan_stage1_18225_runs !=
                 0 ? ctx->scan_stage1_18225_runs : 1),
                (long) ctx->scan_stage1_18225_total_runtime);
        ctx->total_runtime += ctx->scan_stage1_18225_total_runtime;
        ctx->total_runs += ctx->scan_stage1_18225_runs;
        fprintf(stderr,
                "scan_stage2_17841            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage2_17841_runs,
                (long) ctx->scan_stage2_17841_total_runtime /
                (ctx->scan_stage2_17841_runs !=
                 0 ? ctx->scan_stage2_17841_runs : 1),
                (long) ctx->scan_stage2_17841_total_runtime);
        ctx->total_runtime += ctx->scan_stage2_17841_total_runtime;
        ctx->total_runs += ctx->scan_stage2_17841_runs;
        fprintf(stderr,
                "scan_stage2_17862            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage2_17862_runs,
                (long) ctx->scan_stage2_17862_total_runtime /
                (ctx->scan_stage2_17862_runs !=
                 0 ? ctx->scan_stage2_17862_runs : 1),
                (long) ctx->scan_stage2_17862_total_runtime);
        ctx->total_runtime += ctx->scan_stage2_17862_total_runtime;
        ctx->total_runs += ctx->scan_stage2_17862_runs;
        fprintf(stderr,
                "scan_stage2_17871            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage2_17871_runs,
                (long) ctx->scan_stage2_17871_total_runtime /
                (ctx->scan_stage2_17871_runs !=
                 0 ? ctx->scan_stage2_17871_runs : 1),
                (long) ctx->scan_stage2_17871_total_runtime);
        ctx->total_runtime += ctx->scan_stage2_17871_total_runtime;
        ctx->total_runs += ctx->scan_stage2_17871_runs;
        fprintf(stderr,
                "scan_stage2_17960            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage2_17960_runs,
                (long) ctx->scan_stage2_17960_total_runtime /
                (ctx->scan_stage2_17960_runs !=
                 0 ? ctx->scan_stage2_17960_runs : 1),
                (long) ctx->scan_stage2_17960_total_runtime);
        ctx->total_runtime += ctx->scan_stage2_17960_total_runtime;
        ctx->total_runs += ctx->scan_stage2_17960_runs;
        fprintf(stderr,
                "scan_stage2_17969            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage2_17969_runs,
                (long) ctx->scan_stage2_17969_total_runtime /
                (ctx->scan_stage2_17969_runs !=
                 0 ? ctx->scan_stage2_17969_runs : 1),
                (long) ctx->scan_stage2_17969_total_runtime);
        ctx->total_runtime += ctx->scan_stage2_17969_total_runtime;
        ctx->total_runs += ctx->scan_stage2_17969_runs;
        fprintf(stderr,
                "scan_stage2_18044            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage2_18044_runs,
                (long) ctx->scan_stage2_18044_total_runtime /
                (ctx->scan_stage2_18044_runs !=
                 0 ? ctx->scan_stage2_18044_runs : 1),
                (long) ctx->scan_stage2_18044_total_runtime);
        ctx->total_runtime += ctx->scan_stage2_18044_total_runtime;
        ctx->total_runs += ctx->scan_stage2_18044_runs;
        fprintf(stderr,
                "scan_stage2_18065            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage2_18065_runs,
                (long) ctx->scan_stage2_18065_total_runtime /
                (ctx->scan_stage2_18065_runs !=
                 0 ? ctx->scan_stage2_18065_runs : 1),
                (long) ctx->scan_stage2_18065_total_runtime);
        ctx->total_runtime += ctx->scan_stage2_18065_total_runtime;
        ctx->total_runs += ctx->scan_stage2_18065_runs;
        fprintf(stderr,
                "scan_stage2_18225            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage2_18225_runs,
                (long) ctx->scan_stage2_18225_total_runtime /
                (ctx->scan_stage2_18225_runs !=
                 0 ? ctx->scan_stage2_18225_runs : 1),
                (long) ctx->scan_stage2_18225_total_runtime);
        ctx->total_runtime += ctx->scan_stage2_18225_total_runtime;
        ctx->total_runs += ctx->scan_stage2_18225_runs;
        fprintf(stderr,
                "scan_stage3_18569            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage3_18569_runs,
                (long) ctx->scan_stage3_18569_total_runtime /
                (ctx->scan_stage3_18569_runs !=
                 0 ? ctx->scan_stage3_18569_runs : 1),
                (long) ctx->scan_stage3_18569_total_runtime);
        ctx->total_runtime += ctx->scan_stage3_18569_total_runtime;
        ctx->total_runs += ctx->scan_stage3_18569_runs;
        fprintf(stderr,
                "scan_stage3_18649            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage3_18649_runs,
                (long) ctx->scan_stage3_18649_total_runtime /
                (ctx->scan_stage3_18649_runs !=
                 0 ? ctx->scan_stage3_18649_runs : 1),
                (long) ctx->scan_stage3_18649_total_runtime);
        ctx->total_runtime += ctx->scan_stage3_18649_total_runtime;
        ctx->total_runs += ctx->scan_stage3_18649_runs;
        fprintf(stderr,
                "scan_stage3_18695            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage3_18695_runs,
                (long) ctx->scan_stage3_18695_total_runtime /
                (ctx->scan_stage3_18695_runs !=
                 0 ? ctx->scan_stage3_18695_runs : 1),
                (long) ctx->scan_stage3_18695_total_runtime);
        ctx->total_runtime += ctx->scan_stage3_18695_total_runtime;
        ctx->total_runs += ctx->scan_stage3_18695_runs;
        fprintf(stderr,
                "scan_stage3_18876            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage3_18876_runs,
                (long) ctx->scan_stage3_18876_total_runtime /
                (ctx->scan_stage3_18876_runs !=
                 0 ? ctx->scan_stage3_18876_runs : 1),
                (long) ctx->scan_stage3_18876_total_runtime);
        ctx->total_runtime += ctx->scan_stage3_18876_total_runtime;
        ctx->total_runs += ctx->scan_stage3_18876_runs;
        fprintf(stderr,
                "scan_stage3_18922            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage3_18922_runs,
                (long) ctx->scan_stage3_18922_total_runtime /
                (ctx->scan_stage3_18922_runs !=
                 0 ? ctx->scan_stage3_18922_runs : 1),
                (long) ctx->scan_stage3_18922_total_runtime);
        ctx->total_runtime += ctx->scan_stage3_18922_total_runtime;
        ctx->total_runs += ctx->scan_stage3_18922_runs;
        fprintf(stderr,
                "scan_stage3_19001            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage3_19001_runs,
                (long) ctx->scan_stage3_19001_total_runtime /
                (ctx->scan_stage3_19001_runs !=
                 0 ? ctx->scan_stage3_19001_runs : 1),
                (long) ctx->scan_stage3_19001_total_runtime);
        ctx->total_runtime += ctx->scan_stage3_19001_total_runtime;
        ctx->total_runs += ctx->scan_stage3_19001_runs;
        fprintf(stderr,
                "scan_stage3_19086            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage3_19086_runs,
                (long) ctx->scan_stage3_19086_total_runtime /
                (ctx->scan_stage3_19086_runs !=
                 0 ? ctx->scan_stage3_19086_runs : 1),
                (long) ctx->scan_stage3_19086_total_runtime);
        ctx->total_runtime += ctx->scan_stage3_19086_total_runtime;
        ctx->total_runs += ctx->scan_stage3_19086_runs;
        fprintf(stderr,
                "scan_stage3_19155            ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->scan_stage3_19155_runs,
                (long) ctx->scan_stage3_19155_total_runtime /
                (ctx->scan_stage3_19155_runs !=
                 0 ? ctx->scan_stage3_19155_runs : 1),
                (long) ctx->scan_stage3_19155_total_runtime);
        ctx->total_runtime += ctx->scan_stage3_19155_total_runtime;
        ctx->total_runs += ctx->scan_stage3_19155_runs;
        fprintf(stderr,
                "segmap_17843                 ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_17843_runs, (long) ctx->segmap_17843_total_runtime /
                (ctx->segmap_17843_runs != 0 ? ctx->segmap_17843_runs : 1),
                (long) ctx->segmap_17843_total_runtime);
        ctx->total_runtime += ctx->segmap_17843_total_runtime;
        ctx->total_runs += ctx->segmap_17843_runs;
        fprintf(stderr,
                "segmap_17873                 ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_17873_runs, (long) ctx->segmap_17873_total_runtime /
                (ctx->segmap_17873_runs != 0 ? ctx->segmap_17873_runs : 1),
                (long) ctx->segmap_17873_total_runtime);
        ctx->total_runtime += ctx->segmap_17873_total_runtime;
        ctx->total_runs += ctx->segmap_17873_runs;
        fprintf(stderr,
                "segmap_17920                 ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_17920_runs, (long) ctx->segmap_17920_total_runtime /
                (ctx->segmap_17920_runs != 0 ? ctx->segmap_17920_runs : 1),
                (long) ctx->segmap_17920_total_runtime);
        ctx->total_runtime += ctx->segmap_17920_total_runtime;
        ctx->total_runs += ctx->segmap_17920_runs;
        fprintf(stderr,
                "segmap_18005                 ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_18005_runs, (long) ctx->segmap_18005_total_runtime /
                (ctx->segmap_18005_runs != 0 ? ctx->segmap_18005_runs : 1),
                (long) ctx->segmap_18005_total_runtime);
        ctx->total_runtime += ctx->segmap_18005_total_runtime;
        ctx->total_runs += ctx->segmap_18005_runs;
        fprintf(stderr,
                "segmap_18046                 ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_18046_runs, (long) ctx->segmap_18046_total_runtime /
                (ctx->segmap_18046_runs != 0 ? ctx->segmap_18046_runs : 1),
                (long) ctx->segmap_18046_total_runtime);
        ctx->total_runtime += ctx->segmap_18046_total_runtime;
        ctx->total_runs += ctx->segmap_18046_runs;
        fprintf(stderr,
                "segmap_18107                 ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_18107_runs, (long) ctx->segmap_18107_total_runtime /
                (ctx->segmap_18107_runs != 0 ? ctx->segmap_18107_runs : 1),
                (long) ctx->segmap_18107_total_runtime);
        ctx->total_runtime += ctx->segmap_18107_total_runtime;
        ctx->total_runs += ctx->segmap_18107_runs;
        fprintf(stderr,
                "segmap_18145                 ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_18145_runs, (long) ctx->segmap_18145_total_runtime /
                (ctx->segmap_18145_runs != 0 ? ctx->segmap_18145_runs : 1),
                (long) ctx->segmap_18145_total_runtime);
        ctx->total_runtime += ctx->segmap_18145_total_runtime;
        ctx->total_runs += ctx->segmap_18145_runs;
        fprintf(stderr,
                "segmap_18189                 ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_18189_runs, (long) ctx->segmap_18189_total_runtime /
                (ctx->segmap_18189_runs != 0 ? ctx->segmap_18189_runs : 1),
                (long) ctx->segmap_18189_total_runtime);
        ctx->total_runtime += ctx->segmap_18189_total_runtime;
        ctx->total_runs += ctx->segmap_18189_runs;
        fprintf(stderr,
                "segmap_18227                 ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segmap_18227_runs, (long) ctx->segmap_18227_total_runtime /
                (ctx->segmap_18227_runs != 0 ? ctx->segmap_18227_runs : 1),
                (long) ctx->segmap_18227_total_runtime);
        ctx->total_runtime += ctx->segmap_18227_total_runtime;
        ctx->total_runs += ctx->segmap_18227_runs;
        fprintf(stderr,
                "segred_nonseg_17832          ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segred_nonseg_17832_runs,
                (long) ctx->segred_nonseg_17832_total_runtime /
                (ctx->segred_nonseg_17832_runs !=
                 0 ? ctx->segred_nonseg_17832_runs : 1),
                (long) ctx->segred_nonseg_17832_total_runtime);
        ctx->total_runtime += ctx->segred_nonseg_17832_total_runtime;
        ctx->total_runs += ctx->segred_nonseg_17832_runs;
        fprintf(stderr,
                "segred_nonseg_18248          ran %5d times; avg: %8ldus; total: %8ldus\n",
                ctx->segred_nonseg_18248_runs,
                (long) ctx->segred_nonseg_18248_total_runtime /
                (ctx->segred_nonseg_18248_runs !=
                 0 ? ctx->segred_nonseg_18248_runs : 1),
                (long) ctx->segred_nonseg_18248_total_runtime);
        ctx->total_runtime += ctx->segred_nonseg_18248_total_runtime;
        ctx->total_runs += ctx->segred_nonseg_18248_runs;
        if (ctx->profiling)
            fprintf(stderr, "%d operations with cumulative runtime: %6ldus\n",
                    ctx->total_runs, ctx->total_runtime);
    }
}
static int futrts_main(struct futhark_context *ctx,
                       struct memblock_device *out_mem_p_19193,
                       int32_t *out_out_arrsizze_19194,
                       struct memblock_device arr_mem_18295, int32_t n_17467);
static int futrts__map_transpose_i32(struct futhark_context *ctx,
                                     struct memblock_device destmem_0,
                                     int32_t destoffset_1,
                                     struct memblock_device srcmem_2,
                                     int32_t srcoffset_3, int32_t num_arrays_4,
                                     int32_t x_elems_5, int32_t y_elems_6,
                                     int32_t in_elems_7, int32_t out_elems_8);
static int futrts__replicate_f32(struct futhark_context *ctx,
                                 struct memblock_device mem_18934,
                                 int32_t num_elems_18935, float val_18936);
static int futrts__replicate_i32(struct futhark_context *ctx,
                                 struct memblock_device mem_18466,
                                 int32_t num_elems_18467, int32_t val_18468);
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
                       struct memblock_device *out_mem_p_19193,
                       int32_t *out_out_arrsizze_19194,
                       struct memblock_device arr_mem_18295, int32_t n_17467)
{
    struct memblock_device out_mem_18464;
    
    out_mem_18464.references = NULL;
    
    int32_t out_arrsizze_18465;
    struct memblock_device mem_18298;
    
    mem_18298.references = NULL;
    if (memblock_alloc_device(ctx, &mem_18298, 4, "mem_18298"))
        return 1;
    
    int call_ret_19195 = futrts__replicate_i32(ctx, mem_18298, 1, n_17467);
    
    assert(call_ret_19195 == 0);
    
    int32_t iota_arg_17470 = n_17467 - 1;
    int64_t iota_arg_17822 = sext_i32_i64(iota_arg_17470);
    int32_t segred_group_sizze_17825;
    
    segred_group_sizze_17825 = ctx->sizes.mainzisegred_group_sizze_17824;
    
    int32_t num_groups_17827;
    int32_t max_num_groups_18475;
    
    max_num_groups_18475 = ctx->sizes.mainzisegred_num_groups_17826;
    num_groups_17827 = sext_i64_i32(smax64(1, smin64(squot64(iota_arg_17822 +
                                                             sext_i32_i64(segred_group_sizze_17825) -
                                                             1,
                                                             sext_i32_i64(segred_group_sizze_17825)),
                                                     sext_i32_i64(max_num_groups_18475))));
    
    struct memblock_device mem_18301;
    
    mem_18301.references = NULL;
    if (memblock_alloc_device(ctx, &mem_18301, 1, "mem_18301"))
        return 1;
    
    struct memblock_device counter_mem_18476 = ctx->counter_mem_18476;
    struct memblock_device group_res_arr_mem_18478;
    
    group_res_arr_mem_18478.references = NULL;
    if (memblock_alloc_device(ctx, &group_res_arr_mem_18478, sizeof(bool) *
                              (segred_group_sizze_17825 * num_groups_17827),
                              "group_res_arr_mem_18478"))
        return 1;
    
    int32_t num_threads_18480 = num_groups_17827 * segred_group_sizze_17825;
    
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_17832, 0,
                                            sizeof(bool), NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_17832, 1,
                                            sizeof(bool) *
                                            segred_group_sizze_17825, NULL));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_17832, 2,
                                            sizeof(iota_arg_17470),
                                            &iota_arg_17470));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_17832, 3,
                                            sizeof(num_groups_17827),
                                            &num_groups_17827));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_17832, 4,
                                            sizeof(arr_mem_18295.mem),
                                            &arr_mem_18295.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_17832, 5,
                                            sizeof(mem_18301.mem),
                                            &mem_18301.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_17832, 6,
                                            sizeof(counter_mem_18476.mem),
                                            &counter_mem_18476.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_17832, 7,
                                            sizeof(group_res_arr_mem_18478.mem),
                                            &group_res_arr_mem_18478.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_17832, 8,
                                            sizeof(num_threads_18480),
                                            &num_threads_18480));
    if (1 * (num_groups_17827 * segred_group_sizze_17825) != 0) {
        const size_t global_work_sizze_19197[1] = {num_groups_17827 *
                     segred_group_sizze_17825};
        const size_t local_work_sizze_19201[1] = {segred_group_sizze_17825};
        int64_t time_start_19198 = 0, time_end_19199 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "segred_nonseg_17832");
            fprintf(stderr, "%zu", global_work_sizze_19197[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_19201[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) (0 + sizeof(bool) + sizeof(bool) *
                           segred_group_sizze_17825));
            time_start_19198 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->segred_nonseg_17832,
                                                        1, NULL,
                                                        global_work_sizze_19197,
                                                        local_work_sizze_19201,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->segred_nonseg_17832_runs,
                                                                                                        &ctx->segred_nonseg_17832_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_19199 = get_wall_time();
            
            long time_diff_19200 = time_end_19199 - time_start_19198;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "segred_nonseg_17832",
                    time_diff_19200);
        }
    }
    
    bool read_res_19202;
    
    OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                 mem_18301.mem, CL_TRUE, 0 *
                                                 sizeof(bool), sizeof(bool),
                                                 &read_res_19202, 0, NULL,
                                                 ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                 &ctx->copy_scalar_from_dev_runs,
                                                                                                 &ctx->copy_scalar_from_dev_total_runtime)));
    
    bool res_17473 = read_res_19202;
    
    if (memblock_unref_device(ctx, &mem_18301, "mem_18301") != 0)
        return 1;
    
    bool loop_cond_17482 = !res_17473;
    bool cond_17483 = slt32(0, n_17467);
    int32_t segscan_group_sizze_17836;
    
    segscan_group_sizze_17836 = ctx->sizes.mainzisegscan_group_sizze_17835;
    
    int32_t segmap_group_sizze_17847;
    
    segmap_group_sizze_17847 = ctx->sizes.mainzisegmap_group_sizze_17846;
    
    int64_t segmap_group_sizze_17848 = sext_i32_i64(segmap_group_sizze_17847);
    int64_t y_17849 = segmap_group_sizze_17848 - 1;
    int32_t segscan_group_sizze_17857;
    
    segscan_group_sizze_17857 = ctx->sizes.mainzisegscan_group_sizze_17856;
    
    int32_t segscan_group_sizze_17866;
    
    segscan_group_sizze_17866 = ctx->sizes.mainzisegscan_group_sizze_17865;
    
    int32_t segmap_group_sizze_17877;
    
    segmap_group_sizze_17877 = ctx->sizes.mainzisegmap_group_sizze_17876;
    
    int64_t segmap_group_sizze_17878 = sext_i32_i64(segmap_group_sizze_17877);
    int64_t y_17879 = segmap_group_sizze_17878 - 1;
    int64_t n_17936 = sext_i32_i64(n_17467);
    int32_t segmap_group_sizze_17938;
    
    segmap_group_sizze_17938 = ctx->sizes.mainzisegmap_group_sizze_17923;
    
    int64_t segmap_group_sizze_17939 = sext_i32_i64(segmap_group_sizze_17938);
    int64_t y_17940 = segmap_group_sizze_17939 - 1;
    int64_t x_17941 = n_17936 + y_17940;
    int64_t segmap_usable_groups_64_17943;
    
    if (loop_cond_17482) {
        int64_t x_18270 = squot64(x_17941, segmap_group_sizze_17939);
        
        segmap_usable_groups_64_17943 = x_18270;
    } else {
        segmap_usable_groups_64_17943 = 0;
    }
    
    int32_t segmap_usable_groups_17944 =
            sext_i64_i32(segmap_usable_groups_64_17943);
    int32_t segscan_group_sizze_17955;
    
    segscan_group_sizze_17955 = ctx->sizes.mainzisegscan_group_sizze_17954;
    
    int32_t segscan_group_sizze_17964;
    
    segscan_group_sizze_17964 = ctx->sizes.mainzisegscan_group_sizze_17963;
    
    int32_t segmap_group_sizze_18023;
    
    segmap_group_sizze_18023 = ctx->sizes.mainzisegmap_group_sizze_18008;
    
    int64_t segmap_group_sizze_18024 = sext_i32_i64(segmap_group_sizze_18023);
    int64_t y_18025 = segmap_group_sizze_18024 - 1;
    int32_t segscan_group_sizze_18039;
    
    segscan_group_sizze_18039 = ctx->sizes.mainzisegscan_group_sizze_18038;
    
    int32_t segmap_group_sizze_18050;
    
    segmap_group_sizze_18050 = ctx->sizes.mainzisegmap_group_sizze_18049;
    
    int64_t segmap_group_sizze_18051 = sext_i32_i64(segmap_group_sizze_18050);
    int64_t y_18052 = segmap_group_sizze_18051 - 1;
    int32_t segscan_group_sizze_18060;
    
    segscan_group_sizze_18060 = ctx->sizes.mainzisegscan_group_sizze_18059;
    
    int32_t segmap_group_sizze_18128;
    
    segmap_group_sizze_18128 = ctx->sizes.mainzisegmap_group_sizze_18110;
    
    int64_t segmap_group_sizze_18129 = sext_i32_i64(segmap_group_sizze_18128);
    int64_t y_18130 = segmap_group_sizze_18129 - 1;
    int32_t segmap_group_sizze_18149;
    
    segmap_group_sizze_18149 = ctx->sizes.mainzisegmap_group_sizze_18148;
    
    int64_t segmap_group_sizze_18150 = sext_i32_i64(segmap_group_sizze_18149);
    int64_t y_18151 = segmap_group_sizze_18150 - 1;
    int64_t x_18152 = n_17936 + y_18151;
    int64_t segmap_usable_groups_64_18154;
    
    if (loop_cond_17482) {
        int64_t x_18272 = squot64(x_18152, segmap_group_sizze_18150);
        
        segmap_usable_groups_64_18154 = x_18272;
    } else {
        segmap_usable_groups_64_18154 = 0;
    }
    
    int32_t segmap_usable_groups_18155 =
            sext_i64_i32(segmap_usable_groups_64_18154);
    int32_t segmap_group_sizze_18206;
    
    segmap_group_sizze_18206 = ctx->sizes.mainzisegmap_group_sizze_18192;
    
    int32_t segscan_group_sizze_18220;
    
    segscan_group_sizze_18220 = ctx->sizes.mainzisegscan_group_sizze_18219;
    
    int32_t segmap_group_sizze_18231;
    
    segmap_group_sizze_18231 = ctx->sizes.mainzisegmap_group_sizze_18230;
    
    int64_t segmap_group_sizze_18232 = sext_i32_i64(segmap_group_sizze_18231);
    int64_t y_18233 = segmap_group_sizze_18232 - 1;
    int32_t segred_group_sizze_18241;
    
    segred_group_sizze_18241 = ctx->sizes.mainzisegred_group_sizze_18240;
    
    int32_t num_groups_18243;
    int32_t max_num_groups_18502;
    
    max_num_groups_18502 = ctx->sizes.mainzisegred_num_groups_18242;
    num_groups_18243 = sext_i64_i32(smax64(1, smin64(squot64(iota_arg_17822 +
                                                             sext_i32_i64(segred_group_sizze_18241) -
                                                             1,
                                                             sext_i32_i64(segred_group_sizze_18241)),
                                                     sext_i32_i64(max_num_groups_18502))));
    
    int64_t bytes_18331 = 4 * n_17936;
    struct memblock_device mem_18333;
    
    mem_18333.references = NULL;
    if (memblock_alloc_device(ctx, &mem_18333, bytes_18331, "mem_18333"))
        return 1;
    
    struct memblock_device mem_18335;
    
    mem_18335.references = NULL;
    if (memblock_alloc_device(ctx, &mem_18335, n_17936, "mem_18335"))
        return 1;
    
    struct memblock_device mem_18412;
    
    mem_18412.references = NULL;
    if (memblock_alloc_device(ctx, &mem_18412, 1, "mem_18412"))
        return 1;
    
    int32_t sizze_17484;
    struct memblock_device res_mem_18413;
    
    res_mem_18413.references = NULL;
    
    struct memblock_device res_mem_18414;
    
    res_mem_18414.references = NULL;
    
    bool res_17485;
    bool res_17488;
    int32_t res_17489;
    int32_t sizze_17490;
    bool loop_while_17491;
    bool stop_17494;
    int32_t count_17495;
    struct memblock_device shp_mem_18302;
    
    shp_mem_18302.references = NULL;
    
    struct memblock_device arr_mem_18303;
    
    arr_mem_18303.references = NULL;
    sizze_17490 = 1;
    if (memblock_set_device(ctx, &shp_mem_18302, &mem_18298, "mem_18298") != 0)
        return 1;
    if (memblock_set_device(ctx, &arr_mem_18303, &arr_mem_18295,
                            "arr_mem_18295") != 0)
        return 1;
    loop_while_17491 = loop_cond_17482;
    stop_17494 = res_17473;
    count_17495 = 0;
    while (loop_while_17491) {
        int64_t sizze_17833 = sext_i32_i64(sizze_17490);
        int32_t num_groups_17838;
        int32_t max_num_groups_18511;
        
        max_num_groups_18511 = ctx->sizes.mainzisegscan_num_groups_17837;
        num_groups_17838 = sext_i64_i32(smax64(1, smin64(squot64(sizze_17833 +
                                                                 sext_i32_i64(segscan_group_sizze_17836) -
                                                                 1,
                                                                 sext_i32_i64(segscan_group_sizze_17836)),
                                                         sext_i32_i64(max_num_groups_18511))));
        
        int64_t bytes_18305 = 4 * sizze_17833;
        struct memblock_device mem_18307;
        
        mem_18307.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18307, bytes_18305, "mem_18307"))
            return 1;
        
        struct memblock_device mem_18310;
        
        mem_18310.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18310, bytes_18305, "mem_18310"))
            return 1;
        
        int32_t num_threads_18512 = num_groups_17838 *
                segscan_group_sizze_17836;
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17841, 0,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_17836,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17841, 1,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_17836,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17841, 2,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17841, 3,
                                                sizeof(shp_mem_18302.mem),
                                                &shp_mem_18302.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17841, 4,
                                                sizeof(mem_18307.mem),
                                                &mem_18307.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17841, 5,
                                                sizeof(mem_18310.mem),
                                                &mem_18310.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17841, 6,
                                                sizeof(num_threads_18512),
                                                &num_threads_18512));
        if (1 * (num_groups_17838 * segscan_group_sizze_17836) != 0) {
            const size_t global_work_sizze_19203[1] = {num_groups_17838 *
                         segscan_group_sizze_17836};
            const size_t local_work_sizze_19207[1] =
                         {segscan_group_sizze_17836};
            int64_t time_start_19204 = 0, time_end_19205 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage1_17841");
                fprintf(stderr, "%zu", global_work_sizze_19203[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19207[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) * segscan_group_sizze_17836 +
                               sizeof(int32_t) * segscan_group_sizze_17836));
                time_start_19204 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage1_17841,
                                                            1, NULL,
                                                            global_work_sizze_19203,
                                                            local_work_sizze_19207,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage1_17841_runs,
                                                                                                            &ctx->scan_stage1_17841_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19205 = get_wall_time();
                
                long time_diff_19206 = time_end_19205 - time_start_19204;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage1_17841", time_diff_19206);
            }
        }
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegScan");
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "elems_per_group",
                    (long long) (segscan_group_sizze_17836 *
                                 squot32(sizze_17490 + num_threads_18512 - 1,
                                         num_threads_18512)), '\n');
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17841, 0,
                                                sizeof(int32_t) *
                                                num_groups_17838, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17841, 1,
                                                sizeof(int32_t) *
                                                num_groups_17838, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17841, 2,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17841, 3,
                                                sizeof(num_groups_17838),
                                                &num_groups_17838));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17841, 4,
                                                sizeof(mem_18307.mem),
                                                &mem_18307.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17841, 5,
                                                sizeof(mem_18310.mem),
                                                &mem_18310.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17841, 6,
                                                sizeof(num_threads_18512),
                                                &num_threads_18512));
        if (1 * (1 * num_groups_17838) != 0) {
            const size_t global_work_sizze_19208[1] = {1 * num_groups_17838};
            const size_t local_work_sizze_19212[1] = {num_groups_17838};
            int64_t time_start_19209 = 0, time_end_19210 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage2_17841");
                fprintf(stderr, "%zu", global_work_sizze_19208[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19212[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) * num_groups_17838 +
                               sizeof(int32_t) * num_groups_17838));
                time_start_19209 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage2_17841,
                                                            1, NULL,
                                                            global_work_sizze_19208,
                                                            local_work_sizze_19212,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage2_17841_runs,
                                                                                                            &ctx->scan_stage2_17841_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19210 = get_wall_time();
                
                long time_diff_19211 = time_end_19210 - time_start_19209;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage2_17841", time_diff_19211);
            }
        }
        
        int32_t group_sizze_18572;
        
        group_sizze_18572 = ctx->sizes.mainzigroup_sizze_18572;
        
        int32_t num_groups_18573;
        
        num_groups_18573 = squot32(sizze_17490 +
                                   sext_i32_i32(group_sizze_18572) - 1,
                                   sext_i32_i32(group_sizze_18572));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18569, 0,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18569, 1,
                                                sizeof(mem_18307.mem),
                                                &mem_18307.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18569, 2,
                                                sizeof(mem_18310.mem),
                                                &mem_18310.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18569, 3,
                                                sizeof(num_threads_18512),
                                                &num_threads_18512));
        if (1 * (num_groups_18573 * group_sizze_18572) != 0) {
            const size_t global_work_sizze_19213[1] = {num_groups_18573 *
                         group_sizze_18572};
            const size_t local_work_sizze_19217[1] = {group_sizze_18572};
            int64_t time_start_19214 = 0, time_end_19215 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage3_18569");
                fprintf(stderr, "%zu", global_work_sizze_19213[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19217[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19214 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage3_18569,
                                                            1, NULL,
                                                            global_work_sizze_19213,
                                                            local_work_sizze_19217,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage3_18569_runs,
                                                                                                            &ctx->scan_stage3_18569_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19215 = get_wall_time();
                
                long time_diff_19216 = time_end_19215 - time_start_19214;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage3_18569", time_diff_19216);
            }
        }
        
        int32_t i_17511 = sizze_17490 - 1;
        int32_t read_res_19218;
        
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     mem_18310.mem, CL_TRUE,
                                                     i_17511 * sizeof(int32_t),
                                                     sizeof(int32_t),
                                                     &read_res_19218, 0, NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_scalar_from_dev_runs,
                                                                                                     &ctx->copy_scalar_from_dev_total_runtime)));
        
        int32_t x_17512 = read_res_19218;
        int32_t read_res_19219;
        
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     shp_mem_18302.mem, CL_TRUE,
                                                     i_17511 * sizeof(int32_t),
                                                     sizeof(int32_t),
                                                     &read_res_19219, 0, NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_scalar_from_dev_runs,
                                                                                                     &ctx->copy_scalar_from_dev_total_runtime)));
        
        int32_t y_17513 = read_res_19219;
        int32_t aoa_len_17514 = x_17512 + y_17513;
        int64_t binop_x_18312 = sext_i32_i64(aoa_len_17514);
        int64_t bytes_18311 = 4 * binop_x_18312;
        struct memblock_device mem_18313;
        
        mem_18313.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18313, bytes_18311, "mem_18313"))
            return 1;
        
        int call_ret_19220 = futrts__replicate_i32(ctx, mem_18313,
                                                   aoa_len_17514, 0);
        
        assert(call_ret_19220 == 0);
        
        int64_t x_17850 = sizze_17833 + y_17849;
        int64_t segmap_usable_groups_64_17852 = squot64(x_17850,
                                                        segmap_group_sizze_17848);
        int32_t segmap_usable_groups_17853 =
                sext_i64_i32(segmap_usable_groups_64_17852);
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17843, 0,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17843, 1,
                                                sizeof(aoa_len_17514),
                                                &aoa_len_17514));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17843, 2,
                                                sizeof(shp_mem_18302.mem),
                                                &shp_mem_18302.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17843, 3,
                                                sizeof(mem_18310.mem),
                                                &mem_18310.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17843, 4,
                                                sizeof(mem_18313.mem),
                                                &mem_18313.mem));
        if (1 * (segmap_usable_groups_17853 * segmap_group_sizze_17847) != 0) {
            const size_t global_work_sizze_19221[1] =
                         {segmap_usable_groups_17853 *
                         segmap_group_sizze_17847};
            const size_t local_work_sizze_19225[1] = {segmap_group_sizze_17847};
            int64_t time_start_19222 = 0, time_end_19223 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "segmap_17843");
                fprintf(stderr, "%zu", global_work_sizze_19221[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19225[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19222 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->segmap_17843,
                                                            1, NULL,
                                                            global_work_sizze_19221,
                                                            local_work_sizze_19225,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->segmap_17843_runs,
                                                                                                            &ctx->segmap_17843_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19223 = get_wall_time();
                
                long time_diff_19224 = time_end_19223 - time_start_19222;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_17843",
                        time_diff_19224);
            }
        }
        if (memblock_unref_device(ctx, &mem_18310, "mem_18310") != 0)
            return 1;
        
        int32_t num_groups_17859;
        int32_t max_num_groups_18581;
        
        max_num_groups_18581 = ctx->sizes.mainzisegscan_num_groups_17858;
        num_groups_17859 = sext_i64_i32(smax64(1, smin64(squot64(binop_x_18312 +
                                                                 sext_i32_i64(segscan_group_sizze_17857) -
                                                                 1,
                                                                 sext_i32_i64(segscan_group_sizze_17857)),
                                                         sext_i32_i64(max_num_groups_18581))));
        
        struct memblock_device mem_18317;
        
        mem_18317.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18317, bytes_18311, "mem_18317"))
            return 1;
        
        struct memblock_device mem_18320;
        
        mem_18320.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18320, bytes_18311, "mem_18320"))
            return 1;
        
        int32_t num_threads_18582 = num_groups_17859 *
                segscan_group_sizze_17857;
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17862, 0,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_17857,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17862, 1,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_17857,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17862, 2,
                                                sizeof(aoa_len_17514),
                                                &aoa_len_17514));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17862, 3,
                                                sizeof(mem_18313.mem),
                                                &mem_18313.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17862, 4,
                                                sizeof(mem_18317.mem),
                                                &mem_18317.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17862, 5,
                                                sizeof(mem_18320.mem),
                                                &mem_18320.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17862, 6,
                                                sizeof(num_threads_18582),
                                                &num_threads_18582));
        if (1 * (num_groups_17859 * segscan_group_sizze_17857) != 0) {
            const size_t global_work_sizze_19226[1] = {num_groups_17859 *
                         segscan_group_sizze_17857};
            const size_t local_work_sizze_19230[1] =
                         {segscan_group_sizze_17857};
            int64_t time_start_19227 = 0, time_end_19228 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage1_17862");
                fprintf(stderr, "%zu", global_work_sizze_19226[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19230[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) * segscan_group_sizze_17857 +
                               sizeof(int32_t) * segscan_group_sizze_17857));
                time_start_19227 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage1_17862,
                                                            1, NULL,
                                                            global_work_sizze_19226,
                                                            local_work_sizze_19230,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage1_17862_runs,
                                                                                                            &ctx->scan_stage1_17862_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19228 = get_wall_time();
                
                long time_diff_19229 = time_end_19228 - time_start_19227;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage1_17862", time_diff_19229);
            }
        }
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegScan");
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "elems_per_group",
                    (long long) (segscan_group_sizze_17857 *
                                 squot32(aoa_len_17514 + num_threads_18582 - 1,
                                         num_threads_18582)), '\n');
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17862, 0,
                                                sizeof(int32_t) *
                                                num_groups_17859, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17862, 1,
                                                sizeof(int32_t) *
                                                num_groups_17859, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17862, 2,
                                                sizeof(aoa_len_17514),
                                                &aoa_len_17514));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17862, 3,
                                                sizeof(num_groups_17859),
                                                &num_groups_17859));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17862, 4,
                                                sizeof(mem_18317.mem),
                                                &mem_18317.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17862, 5,
                                                sizeof(mem_18320.mem),
                                                &mem_18320.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17862, 6,
                                                sizeof(num_threads_18582),
                                                &num_threads_18582));
        if (1 * (1 * num_groups_17859) != 0) {
            const size_t global_work_sizze_19231[1] = {1 * num_groups_17859};
            const size_t local_work_sizze_19235[1] = {num_groups_17859};
            int64_t time_start_19232 = 0, time_end_19233 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage2_17862");
                fprintf(stderr, "%zu", global_work_sizze_19231[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19235[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) * num_groups_17859 +
                               sizeof(int32_t) * num_groups_17859));
                time_start_19232 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage2_17862,
                                                            1, NULL,
                                                            global_work_sizze_19231,
                                                            local_work_sizze_19235,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage2_17862_runs,
                                                                                                            &ctx->scan_stage2_17862_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19233 = get_wall_time();
                
                long time_diff_19234 = time_end_19233 - time_start_19232;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage2_17862", time_diff_19234);
            }
        }
        
        int32_t group_sizze_18652;
        
        group_sizze_18652 = ctx->sizes.mainzigroup_sizze_18652;
        
        int32_t num_groups_18653;
        
        num_groups_18653 = squot32(aoa_len_17514 +
                                   sext_i32_i32(group_sizze_18652) - 1,
                                   sext_i32_i32(group_sizze_18652));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18649, 0,
                                                sizeof(aoa_len_17514),
                                                &aoa_len_17514));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18649, 1,
                                                sizeof(mem_18317.mem),
                                                &mem_18317.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18649, 2,
                                                sizeof(mem_18320.mem),
                                                &mem_18320.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18649, 3,
                                                sizeof(num_threads_18582),
                                                &num_threads_18582));
        if (1 * (num_groups_18653 * group_sizze_18652) != 0) {
            const size_t global_work_sizze_19236[1] = {num_groups_18653 *
                         group_sizze_18652};
            const size_t local_work_sizze_19240[1] = {group_sizze_18652};
            int64_t time_start_19237 = 0, time_end_19238 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage3_18649");
                fprintf(stderr, "%zu", global_work_sizze_19236[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19240[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19237 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage3_18649,
                                                            1, NULL,
                                                            global_work_sizze_19236,
                                                            local_work_sizze_19240,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage3_18649_runs,
                                                                                                            &ctx->scan_stage3_18649_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19238 = get_wall_time();
                
                long time_diff_19239 = time_end_19238 - time_start_19237;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage3_18649", time_diff_19239);
            }
        }
        if (memblock_unref_device(ctx, &mem_18313, "mem_18313") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18317, "mem_18317") != 0)
            return 1;
        
        int32_t num_groups_17868;
        int32_t max_num_groups_18656;
        
        max_num_groups_18656 = ctx->sizes.mainzisegscan_num_groups_17867;
        num_groups_17868 = sext_i64_i32(smax64(1, smin64(squot64(sizze_17833 +
                                                                 sext_i32_i64(segscan_group_sizze_17866) -
                                                                 1,
                                                                 sext_i32_i64(segscan_group_sizze_17866)),
                                                         sext_i32_i64(max_num_groups_18656))));
        
        struct memblock_device mem_18324;
        
        mem_18324.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18324, bytes_18305, "mem_18324"))
            return 1;
        
        struct memblock_device mem_18327;
        
        mem_18327.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18327, bytes_18305, "mem_18327"))
            return 1;
        
        int32_t num_threads_18657 = num_groups_17868 *
                segscan_group_sizze_17866;
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17871, 0,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_17866,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17871, 1,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17871, 2,
                                                sizeof(count_17495),
                                                &count_17495));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17871, 3,
                                                sizeof(shp_mem_18302.mem),
                                                &shp_mem_18302.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17871, 4,
                                                sizeof(arr_mem_18303.mem),
                                                &arr_mem_18303.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17871, 5,
                                                sizeof(mem_18307.mem),
                                                &mem_18307.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17871, 6,
                                                sizeof(mem_18324.mem),
                                                &mem_18324.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17871, 7,
                                                sizeof(mem_18327.mem),
                                                &mem_18327.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17871, 8,
                                                sizeof(num_threads_18657),
                                                &num_threads_18657));
        if (1 * (num_groups_17868 * segscan_group_sizze_17866) != 0) {
            const size_t global_work_sizze_19241[1] = {num_groups_17868 *
                         segscan_group_sizze_17866};
            const size_t local_work_sizze_19245[1] =
                         {segscan_group_sizze_17866};
            int64_t time_start_19242 = 0, time_end_19243 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage1_17871");
                fprintf(stderr, "%zu", global_work_sizze_19241[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19245[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) *
                               segscan_group_sizze_17866));
                time_start_19242 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage1_17871,
                                                            1, NULL,
                                                            global_work_sizze_19241,
                                                            local_work_sizze_19245,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage1_17871_runs,
                                                                                                            &ctx->scan_stage1_17871_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19243 = get_wall_time();
                
                long time_diff_19244 = time_end_19243 - time_start_19242;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage1_17871", time_diff_19244);
            }
        }
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegScan");
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "elems_per_group",
                    (long long) (segscan_group_sizze_17866 *
                                 squot32(sizze_17490 + num_threads_18657 - 1,
                                         num_threads_18657)), '\n');
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17871, 0,
                                                sizeof(int32_t) *
                                                num_groups_17868, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17871, 1,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17871, 2,
                                                sizeof(num_groups_17868),
                                                &num_groups_17868));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17871, 3,
                                                sizeof(mem_18324.mem),
                                                &mem_18324.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17871, 4,
                                                sizeof(num_threads_18657),
                                                &num_threads_18657));
        if (1 * (1 * num_groups_17868) != 0) {
            const size_t global_work_sizze_19246[1] = {1 * num_groups_17868};
            const size_t local_work_sizze_19250[1] = {num_groups_17868};
            int64_t time_start_19247 = 0, time_end_19248 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage2_17871");
                fprintf(stderr, "%zu", global_work_sizze_19246[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19250[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) * num_groups_17868));
                time_start_19247 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage2_17871,
                                                            1, NULL,
                                                            global_work_sizze_19246,
                                                            local_work_sizze_19250,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage2_17871_runs,
                                                                                                            &ctx->scan_stage2_17871_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19248 = get_wall_time();
                
                long time_diff_19249 = time_end_19248 - time_start_19247;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage2_17871", time_diff_19249);
            }
        }
        
        int32_t group_sizze_18698;
        
        group_sizze_18698 = ctx->sizes.mainzigroup_sizze_18698;
        
        int32_t num_groups_18699;
        
        num_groups_18699 = squot32(sizze_17490 +
                                   sext_i32_i32(group_sizze_18698) - 1,
                                   sext_i32_i32(group_sizze_18698));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18695, 0,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18695, 1,
                                                sizeof(mem_18324.mem),
                                                &mem_18324.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18695, 2,
                                                sizeof(num_threads_18657),
                                                &num_threads_18657));
        if (1 * (num_groups_18699 * group_sizze_18698) != 0) {
            const size_t global_work_sizze_19251[1] = {num_groups_18699 *
                         group_sizze_18698};
            const size_t local_work_sizze_19255[1] = {group_sizze_18698};
            int64_t time_start_19252 = 0, time_end_19253 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage3_18695");
                fprintf(stderr, "%zu", global_work_sizze_19251[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19255[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19252 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage3_18695,
                                                            1, NULL,
                                                            global_work_sizze_19251,
                                                            local_work_sizze_19255,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage3_18695_runs,
                                                                                                            &ctx->scan_stage3_18695_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19253 = get_wall_time();
                
                long time_diff_19254 = time_end_19253 - time_start_19252;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage3_18695", time_diff_19254);
            }
        }
        if (memblock_unref_device(ctx, &mem_18307, "mem_18307") != 0)
            return 1;
        
        int32_t read_res_19256;
        
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     mem_18324.mem, CL_TRUE,
                                                     i_17511 * sizeof(int32_t),
                                                     sizeof(int32_t),
                                                     &read_res_19256, 0, NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_scalar_from_dev_runs,
                                                                                                     &ctx->copy_scalar_from_dev_total_runtime)));
        
        int32_t x_17581 = read_res_19256;
        int32_t aoa_len_17582 = y_17513 + x_17581;
        int64_t binop_x_18329 = sext_i32_i64(aoa_len_17582);
        int64_t bytes_18328 = 4 * binop_x_18329;
        struct memblock_device mem_18330;
        
        mem_18330.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18330, bytes_18328, "mem_18330"))
            return 1;
        
        int call_ret_19257 = futrts__replicate_i32(ctx, mem_18330,
                                                   aoa_len_17582, 0);
        
        assert(call_ret_19257 == 0);
        
        int64_t x_17880 = sizze_17833 + y_17879;
        int64_t segmap_usable_groups_64_17882 = squot64(x_17880,
                                                        segmap_group_sizze_17878);
        int32_t segmap_usable_groups_17883 =
                sext_i64_i32(segmap_usable_groups_64_17882);
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17873, 0,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17873, 1,
                                                sizeof(aoa_len_17582),
                                                &aoa_len_17582));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17873, 2,
                                                sizeof(shp_mem_18302.mem),
                                                &shp_mem_18302.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17873, 3,
                                                sizeof(mem_18324.mem),
                                                &mem_18324.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17873, 4,
                                                sizeof(mem_18330.mem),
                                                &mem_18330.mem));
        if (1 * (segmap_usable_groups_17883 * segmap_group_sizze_17877) != 0) {
            const size_t global_work_sizze_19258[1] =
                         {segmap_usable_groups_17883 *
                         segmap_group_sizze_17877};
            const size_t local_work_sizze_19262[1] = {segmap_group_sizze_17877};
            int64_t time_start_19259 = 0, time_end_19260 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "segmap_17873");
                fprintf(stderr, "%zu", global_work_sizze_19258[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19262[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19259 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->segmap_17873,
                                                            1, NULL,
                                                            global_work_sizze_19258,
                                                            local_work_sizze_19262,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->segmap_17873_runs,
                                                                                                            &ctx->segmap_17873_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19260 = get_wall_time();
                
                long time_diff_19261 = time_end_19260 - time_start_19259;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_17873",
                        time_diff_19261);
            }
        }
        if (memblock_unref_device(ctx, &mem_18324, "mem_18324") != 0)
            return 1;
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17920, 0,
                                                sizeof(n_17467), &n_17467));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17920, 1,
                                                sizeof(arr_mem_18303.mem),
                                                &arr_mem_18303.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17920, 2,
                                                sizeof(mem_18320.mem),
                                                &mem_18320.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17920, 3,
                                                sizeof(mem_18327.mem),
                                                &mem_18327.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17920, 4,
                                                sizeof(mem_18333.mem),
                                                &mem_18333.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_17920, 5,
                                                sizeof(mem_18335.mem),
                                                &mem_18335.mem));
        if (1 * (segmap_usable_groups_17944 * segmap_group_sizze_17938) != 0) {
            const size_t global_work_sizze_19263[1] =
                         {segmap_usable_groups_17944 *
                         segmap_group_sizze_17938};
            const size_t local_work_sizze_19267[1] = {segmap_group_sizze_17938};
            int64_t time_start_19264 = 0, time_end_19265 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "segmap_17920");
                fprintf(stderr, "%zu", global_work_sizze_19263[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19267[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19264 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->segmap_17920,
                                                            1, NULL,
                                                            global_work_sizze_19263,
                                                            local_work_sizze_19267,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->segmap_17920_runs,
                                                                                                            &ctx->segmap_17920_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19265 = get_wall_time();
                
                long time_diff_19266 = time_end_19265 - time_start_19264;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_17920",
                        time_diff_19266);
            }
        }
        if (memblock_unref_device(ctx, &mem_18320, "mem_18320") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18327, "mem_18327") != 0)
            return 1;
        
        int32_t num_groups_17957;
        int32_t max_num_groups_18712;
        
        max_num_groups_18712 = ctx->sizes.mainzisegscan_num_groups_17956;
        num_groups_17957 = sext_i64_i32(smax64(1, smin64(squot64(binop_x_18329 +
                                                                 sext_i32_i64(segscan_group_sizze_17955) -
                                                                 1,
                                                                 sext_i32_i64(segscan_group_sizze_17955)),
                                                         sext_i32_i64(max_num_groups_18712))));
        
        struct memblock_device mem_18339;
        
        mem_18339.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18339, bytes_18328, "mem_18339"))
            return 1;
        
        struct memblock_device mem_18342;
        
        mem_18342.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18342, bytes_18328, "mem_18342"))
            return 1;
        
        struct memblock_device mem_18345;
        
        mem_18345.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18345, bytes_18328, "mem_18345"))
            return 1;
        
        struct memblock_device mem_18348;
        
        mem_18348.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18348, bytes_18328, "mem_18348"))
            return 1;
        
        struct memblock_device mem_18351;
        
        mem_18351.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18351, bytes_18328, "mem_18351"))
            return 1;
        
        struct memblock_device mem_18354;
        
        mem_18354.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18354, bytes_18328, "mem_18354"))
            return 1;
        
        int32_t num_threads_18713 = num_groups_17957 *
                segscan_group_sizze_17955;
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 0,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_17955,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 1,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_17955,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 2,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_17955,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 3,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_17955,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 4,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_17955,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 5,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_17955,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 6,
                                                sizeof(aoa_len_17582),
                                                &aoa_len_17582));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 7,
                                                sizeof(mem_18330.mem),
                                                &mem_18330.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 8,
                                                sizeof(mem_18333.mem),
                                                &mem_18333.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 9,
                                                sizeof(mem_18339.mem),
                                                &mem_18339.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 10,
                                                sizeof(mem_18342.mem),
                                                &mem_18342.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 11,
                                                sizeof(mem_18345.mem),
                                                &mem_18345.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 12,
                                                sizeof(mem_18348.mem),
                                                &mem_18348.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 13,
                                                sizeof(mem_18351.mem),
                                                &mem_18351.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 14,
                                                sizeof(mem_18354.mem),
                                                &mem_18354.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17960, 15,
                                                sizeof(num_threads_18713),
                                                &num_threads_18713));
        if (1 * (num_groups_17957 * segscan_group_sizze_17955) != 0) {
            const size_t global_work_sizze_19268[1] = {num_groups_17957 *
                         segscan_group_sizze_17955};
            const size_t local_work_sizze_19272[1] =
                         {segscan_group_sizze_17955};
            int64_t time_start_19269 = 0, time_end_19270 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage1_17960");
                fprintf(stderr, "%zu", global_work_sizze_19268[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19272[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) * segscan_group_sizze_17955 +
                               sizeof(int32_t) * segscan_group_sizze_17955 +
                               sizeof(int32_t) * segscan_group_sizze_17955 +
                               sizeof(int32_t) * segscan_group_sizze_17955 +
                               sizeof(int32_t) * segscan_group_sizze_17955 +
                               sizeof(int32_t) * segscan_group_sizze_17955));
                time_start_19269 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage1_17960,
                                                            1, NULL,
                                                            global_work_sizze_19268,
                                                            local_work_sizze_19272,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage1_17960_runs,
                                                                                                            &ctx->scan_stage1_17960_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19270 = get_wall_time();
                
                long time_diff_19271 = time_end_19270 - time_start_19269;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage1_17960", time_diff_19271);
            }
        }
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegScan");
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "elems_per_group",
                    (long long) (segscan_group_sizze_17955 *
                                 squot32(aoa_len_17582 + num_threads_18713 - 1,
                                         num_threads_18713)), '\n');
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 0,
                                                sizeof(int32_t) *
                                                num_groups_17957, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 1,
                                                sizeof(int32_t) *
                                                num_groups_17957, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 2,
                                                sizeof(int32_t) *
                                                num_groups_17957, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 3,
                                                sizeof(int32_t) *
                                                num_groups_17957, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 4,
                                                sizeof(int32_t) *
                                                num_groups_17957, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 5,
                                                sizeof(int32_t) *
                                                num_groups_17957, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 6,
                                                sizeof(aoa_len_17582),
                                                &aoa_len_17582));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 7,
                                                sizeof(num_groups_17957),
                                                &num_groups_17957));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 8,
                                                sizeof(mem_18339.mem),
                                                &mem_18339.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 9,
                                                sizeof(mem_18342.mem),
                                                &mem_18342.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 10,
                                                sizeof(mem_18345.mem),
                                                &mem_18345.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 11,
                                                sizeof(mem_18348.mem),
                                                &mem_18348.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 12,
                                                sizeof(mem_18351.mem),
                                                &mem_18351.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 13,
                                                sizeof(mem_18354.mem),
                                                &mem_18354.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17960, 14,
                                                sizeof(num_threads_18713),
                                                &num_threads_18713));
        if (1 * (1 * num_groups_17957) != 0) {
            const size_t global_work_sizze_19273[1] = {1 * num_groups_17957};
            const size_t local_work_sizze_19277[1] = {num_groups_17957};
            int64_t time_start_19274 = 0, time_end_19275 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage2_17960");
                fprintf(stderr, "%zu", global_work_sizze_19273[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19277[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) * num_groups_17957 +
                               sizeof(int32_t) * num_groups_17957 +
                               sizeof(int32_t) * num_groups_17957 +
                               sizeof(int32_t) * num_groups_17957 +
                               sizeof(int32_t) * num_groups_17957 +
                               sizeof(int32_t) * num_groups_17957));
                time_start_19274 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage2_17960,
                                                            1, NULL,
                                                            global_work_sizze_19273,
                                                            local_work_sizze_19277,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage2_17960_runs,
                                                                                                            &ctx->scan_stage2_17960_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19275 = get_wall_time();
                
                long time_diff_19276 = time_end_19275 - time_start_19274;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage2_17960", time_diff_19276);
            }
        }
        
        int32_t group_sizze_18879;
        
        group_sizze_18879 = ctx->sizes.mainzigroup_sizze_18879;
        
        int32_t num_groups_18880;
        
        num_groups_18880 = squot32(aoa_len_17582 +
                                   sext_i32_i32(group_sizze_18879) - 1,
                                   sext_i32_i32(group_sizze_18879));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18876, 0,
                                                sizeof(aoa_len_17582),
                                                &aoa_len_17582));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18876, 1,
                                                sizeof(mem_18339.mem),
                                                &mem_18339.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18876, 2,
                                                sizeof(mem_18342.mem),
                                                &mem_18342.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18876, 3,
                                                sizeof(mem_18345.mem),
                                                &mem_18345.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18876, 4,
                                                sizeof(mem_18348.mem),
                                                &mem_18348.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18876, 5,
                                                sizeof(mem_18351.mem),
                                                &mem_18351.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18876, 6,
                                                sizeof(mem_18354.mem),
                                                &mem_18354.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18876, 7,
                                                sizeof(num_threads_18713),
                                                &num_threads_18713));
        if (1 * (num_groups_18880 * group_sizze_18879) != 0) {
            const size_t global_work_sizze_19278[1] = {num_groups_18880 *
                         group_sizze_18879};
            const size_t local_work_sizze_19282[1] = {group_sizze_18879};
            int64_t time_start_19279 = 0, time_end_19280 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage3_18876");
                fprintf(stderr, "%zu", global_work_sizze_19278[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19282[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19279 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage3_18876,
                                                            1, NULL,
                                                            global_work_sizze_19278,
                                                            local_work_sizze_19282,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage3_18876_runs,
                                                                                                            &ctx->scan_stage3_18876_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19280 = get_wall_time();
                
                long time_diff_19281 = time_end_19280 - time_start_19279;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage3_18876", time_diff_19281);
            }
        }
        if (memblock_unref_device(ctx, &mem_18330, "mem_18330") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18339, "mem_18339") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18345, "mem_18345") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18351, "mem_18351") != 0)
            return 1;
        
        int32_t num_groups_17966;
        int32_t max_num_groups_18883;
        
        max_num_groups_18883 = ctx->sizes.mainzisegscan_num_groups_17965;
        num_groups_17966 = sext_i64_i32(smax64(1, smin64(squot64(sizze_17833 +
                                                                 sext_i32_i64(segscan_group_sizze_17964) -
                                                                 1,
                                                                 sext_i32_i64(segscan_group_sizze_17964)),
                                                         sext_i32_i64(max_num_groups_18883))));
        
        struct memblock_device mem_18358;
        
        mem_18358.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18358, bytes_18305, "mem_18358"))
            return 1;
        
        int32_t num_threads_18884 = num_groups_17966 *
                segscan_group_sizze_17964;
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17969, 0,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_17964,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17969, 1,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17969, 2,
                                                sizeof(shp_mem_18302.mem),
                                                &shp_mem_18302.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17969, 3,
                                                sizeof(mem_18358.mem),
                                                &mem_18358.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_17969, 4,
                                                sizeof(num_threads_18884),
                                                &num_threads_18884));
        if (1 * (num_groups_17966 * segscan_group_sizze_17964) != 0) {
            const size_t global_work_sizze_19283[1] = {num_groups_17966 *
                         segscan_group_sizze_17964};
            const size_t local_work_sizze_19287[1] =
                         {segscan_group_sizze_17964};
            int64_t time_start_19284 = 0, time_end_19285 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage1_17969");
                fprintf(stderr, "%zu", global_work_sizze_19283[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19287[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) *
                               segscan_group_sizze_17964));
                time_start_19284 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage1_17969,
                                                            1, NULL,
                                                            global_work_sizze_19283,
                                                            local_work_sizze_19287,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage1_17969_runs,
                                                                                                            &ctx->scan_stage1_17969_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19285 = get_wall_time();
                
                long time_diff_19286 = time_end_19285 - time_start_19284;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage1_17969", time_diff_19286);
            }
        }
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegScan");
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "elems_per_group",
                    (long long) (segscan_group_sizze_17964 *
                                 squot32(sizze_17490 + num_threads_18884 - 1,
                                         num_threads_18884)), '\n');
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17969, 0,
                                                sizeof(int32_t) *
                                                num_groups_17966, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17969, 1,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17969, 2,
                                                sizeof(num_groups_17966),
                                                &num_groups_17966));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17969, 3,
                                                sizeof(mem_18358.mem),
                                                &mem_18358.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_17969, 4,
                                                sizeof(num_threads_18884),
                                                &num_threads_18884));
        if (1 * (1 * num_groups_17966) != 0) {
            const size_t global_work_sizze_19288[1] = {1 * num_groups_17966};
            const size_t local_work_sizze_19292[1] = {num_groups_17966};
            int64_t time_start_19289 = 0, time_end_19290 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage2_17969");
                fprintf(stderr, "%zu", global_work_sizze_19288[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19292[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) * num_groups_17966));
                time_start_19289 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage2_17969,
                                                            1, NULL,
                                                            global_work_sizze_19288,
                                                            local_work_sizze_19292,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage2_17969_runs,
                                                                                                            &ctx->scan_stage2_17969_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19290 = get_wall_time();
                
                long time_diff_19291 = time_end_19290 - time_start_19289;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage2_17969", time_diff_19291);
            }
        }
        
        int32_t group_sizze_18925;
        
        group_sizze_18925 = ctx->sizes.mainzigroup_sizze_18925;
        
        int32_t num_groups_18926;
        
        num_groups_18926 = squot32(sizze_17490 +
                                   sext_i32_i32(group_sizze_18925) - 1,
                                   sext_i32_i32(group_sizze_18925));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18922, 0,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18922, 1,
                                                sizeof(mem_18358.mem),
                                                &mem_18358.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_18922, 2,
                                                sizeof(num_threads_18884),
                                                &num_threads_18884));
        if (1 * (num_groups_18926 * group_sizze_18925) != 0) {
            const size_t global_work_sizze_19293[1] = {num_groups_18926 *
                         group_sizze_18925};
            const size_t local_work_sizze_19297[1] = {group_sizze_18925};
            int64_t time_start_19294 = 0, time_end_19295 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage3_18922");
                fprintf(stderr, "%zu", global_work_sizze_19293[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19297[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19294 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage3_18922,
                                                            1, NULL,
                                                            global_work_sizze_19293,
                                                            local_work_sizze_19297,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage3_18922_runs,
                                                                                                            &ctx->scan_stage3_18922_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19295 = get_wall_time();
                
                long time_diff_19296 = time_end_19295 - time_start_19294;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage3_18922", time_diff_19296);
            }
        }
        
        int64_t x_18026 = sizze_17833 + y_18025;
        int64_t segmap_usable_groups_64_18028 = squot64(x_18026,
                                                        segmap_group_sizze_18024);
        int32_t segmap_usable_groups_18029 =
                sext_i64_i32(segmap_usable_groups_64_18028);
        struct memblock_device mem_18361;
        
        mem_18361.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18361, bytes_18305, "mem_18361"))
            return 1;
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18005, 0,
                                                sizeof(cond_17483),
                                                &cond_17483));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18005, 1,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18005, 2,
                                                sizeof(mem_18348.mem),
                                                &mem_18348.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18005, 3,
                                                sizeof(mem_18358.mem),
                                                &mem_18358.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18005, 4,
                                                sizeof(mem_18361.mem),
                                                &mem_18361.mem));
        if (1 * (segmap_usable_groups_18029 * segmap_group_sizze_18023) != 0) {
            const size_t global_work_sizze_19298[1] =
                         {segmap_usable_groups_18029 *
                         segmap_group_sizze_18023};
            const size_t local_work_sizze_19302[1] = {segmap_group_sizze_18023};
            int64_t time_start_19299 = 0, time_end_19300 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "segmap_18005");
                fprintf(stderr, "%zu", global_work_sizze_19298[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19302[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19299 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->segmap_18005,
                                                            1, NULL,
                                                            global_work_sizze_19298,
                                                            local_work_sizze_19302,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->segmap_18005_runs,
                                                                                                            &ctx->segmap_18005_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19300 = get_wall_time();
                
                long time_diff_19301 = time_end_19300 - time_start_19299;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_18005",
                        time_diff_19301);
            }
        }
        if (memblock_unref_device(ctx, &mem_18358, "mem_18358") != 0)
            return 1;
        
        struct memblock_device mem_18364;
        
        mem_18364.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18364, bytes_18331, "mem_18364"))
            return 1;
        
        int call_ret_19303 = futrts__replicate_f32(ctx, mem_18364, n_17467,
                                                   0.0F);
        
        assert(call_ret_19303 == 0);
        
        int32_t num_groups_18041;
        int32_t max_num_groups_18943;
        
        max_num_groups_18943 = ctx->sizes.mainzisegscan_num_groups_18040;
        num_groups_18041 = sext_i64_i32(smax64(1, smin64(squot64(sizze_17833 +
                                                                 sext_i32_i64(segscan_group_sizze_18039) -
                                                                 1,
                                                                 sext_i32_i64(segscan_group_sizze_18039)),
                                                         sext_i32_i64(max_num_groups_18943))));
        
        struct memblock_device mem_18368;
        
        mem_18368.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18368, bytes_18305, "mem_18368"))
            return 1;
        
        struct memblock_device mem_18371;
        
        mem_18371.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18371, bytes_18305, "mem_18371"))
            return 1;
        
        int32_t num_threads_18944 = num_groups_18041 *
                segscan_group_sizze_18039;
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18044, 0,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_18039,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18044, 1,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_18039,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18044, 2,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18044, 3,
                                                sizeof(shp_mem_18302.mem),
                                                &shp_mem_18302.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18044, 4,
                                                sizeof(mem_18368.mem),
                                                &mem_18368.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18044, 5,
                                                sizeof(mem_18371.mem),
                                                &mem_18371.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18044, 6,
                                                sizeof(num_threads_18944),
                                                &num_threads_18944));
        if (1 * (num_groups_18041 * segscan_group_sizze_18039) != 0) {
            const size_t global_work_sizze_19304[1] = {num_groups_18041 *
                         segscan_group_sizze_18039};
            const size_t local_work_sizze_19308[1] =
                         {segscan_group_sizze_18039};
            int64_t time_start_19305 = 0, time_end_19306 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage1_18044");
                fprintf(stderr, "%zu", global_work_sizze_19304[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19308[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) * segscan_group_sizze_18039 +
                               sizeof(int32_t) * segscan_group_sizze_18039));
                time_start_19305 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage1_18044,
                                                            1, NULL,
                                                            global_work_sizze_19304,
                                                            local_work_sizze_19308,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage1_18044_runs,
                                                                                                            &ctx->scan_stage1_18044_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19306 = get_wall_time();
                
                long time_diff_19307 = time_end_19306 - time_start_19305;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage1_18044", time_diff_19307);
            }
        }
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegScan");
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "elems_per_group",
                    (long long) (segscan_group_sizze_18039 *
                                 squot32(sizze_17490 + num_threads_18944 - 1,
                                         num_threads_18944)), '\n');
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18044, 0,
                                                sizeof(int32_t) *
                                                num_groups_18041, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18044, 1,
                                                sizeof(int32_t) *
                                                num_groups_18041, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18044, 2,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18044, 3,
                                                sizeof(num_groups_18041),
                                                &num_groups_18041));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18044, 4,
                                                sizeof(mem_18368.mem),
                                                &mem_18368.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18044, 5,
                                                sizeof(mem_18371.mem),
                                                &mem_18371.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18044, 6,
                                                sizeof(num_threads_18944),
                                                &num_threads_18944));
        if (1 * (1 * num_groups_18041) != 0) {
            const size_t global_work_sizze_19309[1] = {1 * num_groups_18041};
            const size_t local_work_sizze_19313[1] = {num_groups_18041};
            int64_t time_start_19310 = 0, time_end_19311 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage2_18044");
                fprintf(stderr, "%zu", global_work_sizze_19309[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19313[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) * num_groups_18041 +
                               sizeof(int32_t) * num_groups_18041));
                time_start_19310 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage2_18044,
                                                            1, NULL,
                                                            global_work_sizze_19309,
                                                            local_work_sizze_19313,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage2_18044_runs,
                                                                                                            &ctx->scan_stage2_18044_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19311 = get_wall_time();
                
                long time_diff_19312 = time_end_19311 - time_start_19310;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage2_18044", time_diff_19312);
            }
        }
        
        int32_t group_sizze_19004;
        
        group_sizze_19004 = ctx->sizes.mainzigroup_sizze_19004;
        
        int32_t num_groups_19005;
        
        num_groups_19005 = squot32(sizze_17490 +
                                   sext_i32_i32(group_sizze_19004) - 1,
                                   sext_i32_i32(group_sizze_19004));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_19001, 0,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_19001, 1,
                                                sizeof(mem_18368.mem),
                                                &mem_18368.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_19001, 2,
                                                sizeof(mem_18371.mem),
                                                &mem_18371.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_19001, 3,
                                                sizeof(num_threads_18944),
                                                &num_threads_18944));
        if (1 * (num_groups_19005 * group_sizze_19004) != 0) {
            const size_t global_work_sizze_19314[1] = {num_groups_19005 *
                         group_sizze_19004};
            const size_t local_work_sizze_19318[1] = {group_sizze_19004};
            int64_t time_start_19315 = 0, time_end_19316 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage3_19001");
                fprintf(stderr, "%zu", global_work_sizze_19314[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19318[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19315 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage3_19001,
                                                            1, NULL,
                                                            global_work_sizze_19314,
                                                            local_work_sizze_19318,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage3_19001_runs,
                                                                                                            &ctx->scan_stage3_19001_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19316 = get_wall_time();
                
                long time_diff_19317 = time_end_19316 - time_start_19315;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage3_19001", time_diff_19317);
            }
        }
        
        int32_t read_res_19319;
        
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     mem_18371.mem, CL_TRUE,
                                                     i_17511 * sizeof(int32_t),
                                                     sizeof(int32_t),
                                                     &read_res_19319, 0, NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_scalar_from_dev_runs,
                                                                                                     &ctx->copy_scalar_from_dev_total_runtime)));
        
        int32_t x_17710 = read_res_19319;
        int32_t aoa_len_17711 = y_17513 + x_17710;
        int64_t binop_x_18373 = sext_i32_i64(aoa_len_17711);
        int64_t bytes_18372 = 4 * binop_x_18373;
        struct memblock_device mem_18374;
        
        mem_18374.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18374, bytes_18372, "mem_18374"))
            return 1;
        
        int call_ret_19320 = futrts__replicate_i32(ctx, mem_18374,
                                                   aoa_len_17711, 0);
        
        assert(call_ret_19320 == 0);
        
        int64_t x_18053 = sizze_17833 + y_18052;
        int64_t segmap_usable_groups_64_18055 = squot64(x_18053,
                                                        segmap_group_sizze_18051);
        int32_t segmap_usable_groups_18056 =
                sext_i64_i32(segmap_usable_groups_64_18055);
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18046, 0,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18046, 1,
                                                sizeof(aoa_len_17711),
                                                &aoa_len_17711));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18046, 2,
                                                sizeof(shp_mem_18302.mem),
                                                &shp_mem_18302.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18046, 3,
                                                sizeof(mem_18371.mem),
                                                &mem_18371.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18046, 4,
                                                sizeof(mem_18374.mem),
                                                &mem_18374.mem));
        if (1 * (segmap_usable_groups_18056 * segmap_group_sizze_18050) != 0) {
            const size_t global_work_sizze_19321[1] =
                         {segmap_usable_groups_18056 *
                         segmap_group_sizze_18050};
            const size_t local_work_sizze_19325[1] = {segmap_group_sizze_18050};
            int64_t time_start_19322 = 0, time_end_19323 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "segmap_18046");
                fprintf(stderr, "%zu", global_work_sizze_19321[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19325[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19322 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->segmap_18046,
                                                            1, NULL,
                                                            global_work_sizze_19321,
                                                            local_work_sizze_19325,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->segmap_18046_runs,
                                                                                                            &ctx->segmap_18046_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19323 = get_wall_time();
                
                long time_diff_19324 = time_end_19323 - time_start_19322;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_18046",
                        time_diff_19324);
            }
        }
        if (memblock_unref_device(ctx, &mem_18371, "mem_18371") != 0)
            return 1;
        
        int32_t num_groups_18062;
        int32_t max_num_groups_19013;
        
        max_num_groups_19013 = ctx->sizes.mainzisegscan_num_groups_18061;
        num_groups_18062 = sext_i64_i32(smax64(1, smin64(squot64(binop_x_18373 +
                                                                 sext_i32_i64(segscan_group_sizze_18060) -
                                                                 1,
                                                                 sext_i32_i64(segscan_group_sizze_18060)),
                                                         sext_i32_i64(max_num_groups_19013))));
        
        struct memblock_device mem_18378;
        
        mem_18378.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18378, bytes_18372, "mem_18378"))
            return 1;
        
        struct memblock_device mem_18381;
        
        mem_18381.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18381, bytes_18372, "mem_18381"))
            return 1;
        
        int32_t num_threads_19014 = num_groups_18062 *
                segscan_group_sizze_18060;
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18065, 0,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_18060,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18065, 1,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_18060,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18065, 2,
                                                sizeof(aoa_len_17711),
                                                &aoa_len_17711));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18065, 3,
                                                sizeof(mem_18374.mem),
                                                &mem_18374.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18065, 4,
                                                sizeof(mem_18378.mem),
                                                &mem_18378.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18065, 5,
                                                sizeof(mem_18381.mem),
                                                &mem_18381.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18065, 6,
                                                sizeof(num_threads_19014),
                                                &num_threads_19014));
        if (1 * (num_groups_18062 * segscan_group_sizze_18060) != 0) {
            const size_t global_work_sizze_19326[1] = {num_groups_18062 *
                         segscan_group_sizze_18060};
            const size_t local_work_sizze_19330[1] =
                         {segscan_group_sizze_18060};
            int64_t time_start_19327 = 0, time_end_19328 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage1_18065");
                fprintf(stderr, "%zu", global_work_sizze_19326[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19330[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) * segscan_group_sizze_18060 +
                               sizeof(int32_t) * segscan_group_sizze_18060));
                time_start_19327 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage1_18065,
                                                            1, NULL,
                                                            global_work_sizze_19326,
                                                            local_work_sizze_19330,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage1_18065_runs,
                                                                                                            &ctx->scan_stage1_18065_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19328 = get_wall_time();
                
                long time_diff_19329 = time_end_19328 - time_start_19327;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage1_18065", time_diff_19329);
            }
        }
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegScan");
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "elems_per_group",
                    (long long) (segscan_group_sizze_18060 *
                                 squot32(aoa_len_17711 + num_threads_19014 - 1,
                                         num_threads_19014)), '\n');
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18065, 0,
                                                sizeof(int32_t) *
                                                num_groups_18062, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18065, 1,
                                                sizeof(int32_t) *
                                                num_groups_18062, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18065, 2,
                                                sizeof(aoa_len_17711),
                                                &aoa_len_17711));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18065, 3,
                                                sizeof(num_groups_18062),
                                                &num_groups_18062));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18065, 4,
                                                sizeof(mem_18378.mem),
                                                &mem_18378.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18065, 5,
                                                sizeof(mem_18381.mem),
                                                &mem_18381.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18065, 6,
                                                sizeof(num_threads_19014),
                                                &num_threads_19014));
        if (1 * (1 * num_groups_18062) != 0) {
            const size_t global_work_sizze_19331[1] = {1 * num_groups_18062};
            const size_t local_work_sizze_19335[1] = {num_groups_18062};
            int64_t time_start_19332 = 0, time_end_19333 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage2_18065");
                fprintf(stderr, "%zu", global_work_sizze_19331[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19335[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) * num_groups_18062 +
                               sizeof(int32_t) * num_groups_18062));
                time_start_19332 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage2_18065,
                                                            1, NULL,
                                                            global_work_sizze_19331,
                                                            local_work_sizze_19335,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage2_18065_runs,
                                                                                                            &ctx->scan_stage2_18065_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19333 = get_wall_time();
                
                long time_diff_19334 = time_end_19333 - time_start_19332;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage2_18065", time_diff_19334);
            }
        }
        
        int32_t group_sizze_19089;
        
        group_sizze_19089 = ctx->sizes.mainzigroup_sizze_19089;
        
        int32_t num_groups_19090;
        
        num_groups_19090 = squot32(aoa_len_17711 +
                                   sext_i32_i32(group_sizze_19089) - 1,
                                   sext_i32_i32(group_sizze_19089));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_19086, 0,
                                                sizeof(aoa_len_17711),
                                                &aoa_len_17711));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_19086, 1,
                                                sizeof(mem_18378.mem),
                                                &mem_18378.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_19086, 2,
                                                sizeof(mem_18381.mem),
                                                &mem_18381.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_19086, 3,
                                                sizeof(num_threads_19014),
                                                &num_threads_19014));
        if (1 * (num_groups_19090 * group_sizze_19089) != 0) {
            const size_t global_work_sizze_19336[1] = {num_groups_19090 *
                         group_sizze_19089};
            const size_t local_work_sizze_19340[1] = {group_sizze_19089};
            int64_t time_start_19337 = 0, time_end_19338 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage3_19086");
                fprintf(stderr, "%zu", global_work_sizze_19336[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19340[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19337 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage3_19086,
                                                            1, NULL,
                                                            global_work_sizze_19336,
                                                            local_work_sizze_19340,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage3_19086_runs,
                                                                                                            &ctx->scan_stage3_19086_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19338 = get_wall_time();
                
                long time_diff_19339 = time_end_19338 - time_start_19337;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage3_19086", time_diff_19339);
            }
        }
        if (memblock_unref_device(ctx, &mem_18374, "mem_18374") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18378, "mem_18378") != 0)
            return 1;
        
        int64_t x_18131 = y_18130 + binop_x_18373;
        int64_t segmap_usable_groups_64_18133 = squot64(x_18131,
                                                        segmap_group_sizze_18129);
        int32_t segmap_usable_groups_18134 =
                sext_i64_i32(segmap_usable_groups_64_18133);
        struct memblock_device mem_18384;
        
        mem_18384.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18384, bytes_18372, "mem_18384"))
            return 1;
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18107, 0,
                                                sizeof(aoa_len_17711),
                                                &aoa_len_17711));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18107, 1,
                                                sizeof(mem_18368.mem),
                                                &mem_18368.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18107, 2,
                                                sizeof(mem_18381.mem),
                                                &mem_18381.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18107, 3,
                                                sizeof(mem_18384.mem),
                                                &mem_18384.mem));
        if (1 * (segmap_usable_groups_18134 * segmap_group_sizze_18128) != 0) {
            const size_t global_work_sizze_19341[1] =
                         {segmap_usable_groups_18134 *
                         segmap_group_sizze_18128};
            const size_t local_work_sizze_19345[1] = {segmap_group_sizze_18128};
            int64_t time_start_19342 = 0, time_end_19343 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "segmap_18107");
                fprintf(stderr, "%zu", global_work_sizze_19341[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19345[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19342 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->segmap_18107,
                                                            1, NULL,
                                                            global_work_sizze_19341,
                                                            local_work_sizze_19345,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->segmap_18107_runs,
                                                                                                            &ctx->segmap_18107_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19343 = get_wall_time();
                
                long time_diff_19344 = time_end_19343 - time_start_19342;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_18107",
                        time_diff_19344);
            }
        }
        if (memblock_unref_device(ctx, &mem_18368, "mem_18368") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18381, "mem_18381") != 0)
            return 1;
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18145, 0,
                                                sizeof(n_17467), &n_17467));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18145, 1,
                                                sizeof(arr_mem_18303.mem),
                                                &arr_mem_18303.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18145, 2,
                                                sizeof(mem_18335.mem),
                                                &mem_18335.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18145, 3,
                                                sizeof(mem_18342.mem),
                                                &mem_18342.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18145, 4,
                                                sizeof(mem_18348.mem),
                                                &mem_18348.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18145, 5,
                                                sizeof(mem_18354.mem),
                                                &mem_18354.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18145, 6,
                                                sizeof(mem_18361.mem),
                                                &mem_18361.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18145, 7,
                                                sizeof(mem_18364.mem),
                                                &mem_18364.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18145, 8,
                                                sizeof(mem_18384.mem),
                                                &mem_18384.mem));
        if (1 * (segmap_usable_groups_18155 * segmap_group_sizze_18149) != 0) {
            const size_t global_work_sizze_19346[1] =
                         {segmap_usable_groups_18155 *
                         segmap_group_sizze_18149};
            const size_t local_work_sizze_19350[1] = {segmap_group_sizze_18149};
            int64_t time_start_19347 = 0, time_end_19348 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "segmap_18145");
                fprintf(stderr, "%zu", global_work_sizze_19346[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19350[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19347 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->segmap_18145,
                                                            1, NULL,
                                                            global_work_sizze_19346,
                                                            local_work_sizze_19350,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->segmap_18145_runs,
                                                                                                            &ctx->segmap_18145_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19348 = get_wall_time();
                
                long time_diff_19349 = time_end_19348 - time_start_19347;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_18145",
                        time_diff_19349);
            }
        }
        if (memblock_unref_device(ctx, &mem_18342, "mem_18342") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18348, "mem_18348") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18354, "mem_18354") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18361, "mem_18361") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18384, "mem_18384") != 0)
            return 1;
        
        int32_t num_groups_18207;
        int32_t max_num_groups_19103;
        
        max_num_groups_19103 = ctx->sizes.mainzisegmap_num_groups_18194;
        num_groups_18207 = sext_i64_i32(smax64(1, smin64(squot64(sizze_17833 +
                                                                 sext_i32_i64(segmap_group_sizze_18206) -
                                                                 1,
                                                                 sext_i32_i64(segmap_group_sizze_18206)),
                                                         sext_i32_i64(max_num_groups_19103))));
        
        int32_t convop_x_18393 = 2 * sizze_17490;
        int64_t binop_x_18394 = sext_i32_i64(convop_x_18393);
        int64_t bytes_18392 = 4 * binop_x_18394;
        struct memblock_device mem_18395;
        
        mem_18395.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18395, bytes_18392, "mem_18395"))
            return 1;
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18189, 0,
                                                sizeof(sizze_17490),
                                                &sizze_17490));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18189, 1,
                                                sizeof(num_groups_18207),
                                                &num_groups_18207));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18189, 2,
                                                sizeof(shp_mem_18302.mem),
                                                &shp_mem_18302.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18189, 3,
                                                sizeof(mem_18395.mem),
                                                &mem_18395.mem));
        if (1 * (num_groups_18207 * segmap_group_sizze_18206) != 0) {
            const size_t global_work_sizze_19351[1] = {num_groups_18207 *
                         segmap_group_sizze_18206};
            const size_t local_work_sizze_19355[1] = {segmap_group_sizze_18206};
            int64_t time_start_19352 = 0, time_end_19353 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "segmap_18189");
                fprintf(stderr, "%zu", global_work_sizze_19351[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19355[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19352 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->segmap_18189,
                                                            1, NULL,
                                                            global_work_sizze_19351,
                                                            local_work_sizze_19355,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->segmap_18189_runs,
                                                                                                            &ctx->segmap_18189_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19353 = get_wall_time();
                
                long time_diff_19354 = time_end_19353 - time_start_19352;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_18189",
                        time_diff_19354);
            }
        }
        
        int32_t num_groups_18222;
        int32_t max_num_groups_19116;
        
        max_num_groups_19116 = ctx->sizes.mainzisegscan_num_groups_18221;
        num_groups_18222 = sext_i64_i32(smax64(1, smin64(squot64(binop_x_18394 +
                                                                 sext_i32_i64(segscan_group_sizze_18220) -
                                                                 1,
                                                                 sext_i32_i64(segscan_group_sizze_18220)),
                                                         sext_i32_i64(max_num_groups_19116))));
        
        struct memblock_device mem_18399;
        
        mem_18399.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18399, bytes_18392, "mem_18399"))
            return 1;
        
        int call_ret_19356 = futrts__map_transpose_i32(ctx, mem_18399, 0,
                                                       mem_18395, 0, 1,
                                                       sizze_17490, 2,
                                                       sizze_17490 * 2,
                                                       sizze_17490 * 2);
        
        assert(call_ret_19356 == 0);
        if (memblock_unref_device(ctx, &mem_18395, "mem_18395") != 0)
            return 1;
        
        struct memblock_device mem_18403;
        
        mem_18403.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18403, bytes_18392, "mem_18403"))
            return 1;
        
        struct memblock_device mem_18406;
        
        mem_18406.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18406, bytes_18392, "mem_18406"))
            return 1;
        
        int32_t num_threads_19117 = num_groups_18222 *
                segscan_group_sizze_18220;
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18225, 0,
                                                sizeof(int32_t) *
                                                segscan_group_sizze_18220,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18225, 1,
                                                sizeof(convop_x_18393),
                                                &convop_x_18393));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18225, 2,
                                                sizeof(mem_18399.mem),
                                                &mem_18399.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18225, 3,
                                                sizeof(mem_18403.mem),
                                                &mem_18403.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18225, 4,
                                                sizeof(mem_18406.mem),
                                                &mem_18406.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage1_18225, 5,
                                                sizeof(num_threads_19117),
                                                &num_threads_19117));
        if (1 * (num_groups_18222 * segscan_group_sizze_18220) != 0) {
            const size_t global_work_sizze_19357[1] = {num_groups_18222 *
                         segscan_group_sizze_18220};
            const size_t local_work_sizze_19361[1] =
                         {segscan_group_sizze_18220};
            int64_t time_start_19358 = 0, time_end_19359 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage1_18225");
                fprintf(stderr, "%zu", global_work_sizze_19357[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19361[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) *
                               segscan_group_sizze_18220));
                time_start_19358 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage1_18225,
                                                            1, NULL,
                                                            global_work_sizze_19357,
                                                            local_work_sizze_19361,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage1_18225_runs,
                                                                                                            &ctx->scan_stage1_18225_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19359 = get_wall_time();
                
                long time_diff_19360 = time_end_19359 - time_start_19358;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage1_18225", time_diff_19360);
            }
        }
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegScan");
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "elems_per_group",
                    (long long) (segscan_group_sizze_18220 *
                                 squot32(convop_x_18393 + num_threads_19117 - 1,
                                         num_threads_19117)), '\n');
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18225, 0,
                                                sizeof(int32_t) *
                                                num_groups_18222, NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18225, 1,
                                                sizeof(num_groups_18222),
                                                &num_groups_18222));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18225, 2,
                                                sizeof(convop_x_18393),
                                                &convop_x_18393));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18225, 3,
                                                sizeof(mem_18403.mem),
                                                &mem_18403.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage2_18225, 4,
                                                sizeof(num_threads_19117),
                                                &num_threads_19117));
        if (1 * (1 * num_groups_18222) != 0) {
            const size_t global_work_sizze_19362[1] = {1 * num_groups_18222};
            const size_t local_work_sizze_19366[1] = {num_groups_18222};
            int64_t time_start_19363 = 0, time_end_19364 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage2_18225");
                fprintf(stderr, "%zu", global_work_sizze_19362[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19366[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(int32_t) * num_groups_18222));
                time_start_19363 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage2_18225,
                                                            1, NULL,
                                                            global_work_sizze_19362,
                                                            local_work_sizze_19366,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage2_18225_runs,
                                                                                                            &ctx->scan_stage2_18225_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19364 = get_wall_time();
                
                long time_diff_19365 = time_end_19364 - time_start_19363;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage2_18225", time_diff_19365);
            }
        }
        
        int32_t group_sizze_19158;
        
        group_sizze_19158 = ctx->sizes.mainzigroup_sizze_19158;
        
        int32_t num_groups_19159;
        
        num_groups_19159 = squot32(convop_x_18393 +
                                   sext_i32_i32(group_sizze_19158) - 1,
                                   sext_i32_i32(group_sizze_19158));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_19155, 0,
                                                sizeof(convop_x_18393),
                                                &convop_x_18393));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_19155, 1,
                                                sizeof(mem_18403.mem),
                                                &mem_18403.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->scan_stage3_19155, 2,
                                                sizeof(num_threads_19117),
                                                &num_threads_19117));
        if (1 * (num_groups_19159 * group_sizze_19158) != 0) {
            const size_t global_work_sizze_19367[1] = {num_groups_19159 *
                         group_sizze_19158};
            const size_t local_work_sizze_19371[1] = {group_sizze_19158};
            int64_t time_start_19368 = 0, time_end_19369 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "scan_stage3_19155");
                fprintf(stderr, "%zu", global_work_sizze_19367[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19371[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19368 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->scan_stage3_19155,
                                                            1, NULL,
                                                            global_work_sizze_19367,
                                                            local_work_sizze_19371,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->scan_stage3_19155_runs,
                                                                                                            &ctx->scan_stage3_19155_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19369 = get_wall_time();
                
                long time_diff_19370 = time_end_19369 - time_start_19368;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "scan_stage3_19155", time_diff_19370);
            }
        }
        
        int32_t last_index_17798 = convop_x_18393 - 1;
        bool is_empty_17799 = convop_x_18393 == 0;
        int32_t partition_sizze_17800;
        
        if (is_empty_17799) {
            partition_sizze_17800 = 0;
        } else {
            int32_t read_res_19372;
            
            OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                         mem_18403.mem, CL_TRUE,
                                                         last_index_17798 *
                                                         sizeof(int32_t),
                                                         sizeof(int32_t),
                                                         &read_res_19372, 0,
                                                         NULL,
                                                         ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                         &ctx->copy_scalar_from_dev_runs,
                                                                                                         &ctx->copy_scalar_from_dev_total_runtime)));
            
            int32_t last_offset_17801 = read_res_19372;
            
            partition_sizze_17800 = last_offset_17801;
        }
        
        int64_t binop_x_18408 = sext_i32_i64(partition_sizze_17800);
        int64_t bytes_18407 = 4 * binop_x_18408;
        struct memblock_device mem_18409;
        
        mem_18409.references = NULL;
        if (memblock_alloc_device(ctx, &mem_18409, bytes_18407, "mem_18409"))
            return 1;
        
        int64_t x_18234 = y_18233 + binop_x_18394;
        int64_t segmap_usable_groups_64_18236 = squot64(x_18234,
                                                        segmap_group_sizze_18232);
        int32_t segmap_usable_groups_18237 =
                sext_i64_i32(segmap_usable_groups_64_18236);
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18227, 0,
                                                sizeof(partition_sizze_17800),
                                                &partition_sizze_17800));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18227, 1,
                                                sizeof(convop_x_18393),
                                                &convop_x_18393));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18227, 2,
                                                sizeof(mem_18399.mem),
                                                &mem_18399.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18227, 3,
                                                sizeof(mem_18403.mem),
                                                &mem_18403.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18227, 4,
                                                sizeof(mem_18406.mem),
                                                &mem_18406.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segmap_18227, 5,
                                                sizeof(mem_18409.mem),
                                                &mem_18409.mem));
        if (1 * (segmap_usable_groups_18237 * segmap_group_sizze_18231) != 0) {
            const size_t global_work_sizze_19373[1] =
                         {segmap_usable_groups_18237 *
                         segmap_group_sizze_18231};
            const size_t local_work_sizze_19377[1] = {segmap_group_sizze_18231};
            int64_t time_start_19374 = 0, time_end_19375 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "segmap_18227");
                fprintf(stderr, "%zu", global_work_sizze_19373[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19377[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) 0);
                time_start_19374 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->segmap_18227,
                                                            1, NULL,
                                                            global_work_sizze_19373,
                                                            local_work_sizze_19377,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->segmap_18227_runs,
                                                                                                            &ctx->segmap_18227_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19375 = get_wall_time();
                
                long time_diff_19376 = time_end_19375 - time_start_19374;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n", "segmap_18227",
                        time_diff_19376);
            }
        }
        if (memblock_unref_device(ctx, &mem_18399, "mem_18399") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18403, "mem_18403") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18406, "mem_18406") != 0)
            return 1;
        
        struct memblock_device counter_mem_19167 = ctx->counter_mem_19167;
        struct memblock_device group_res_arr_mem_19169;
        
        group_res_arr_mem_19169.references = NULL;
        if (memblock_alloc_device(ctx, &group_res_arr_mem_19169, sizeof(bool) *
                                  (segred_group_sizze_18241 * num_groups_18243),
                                  "group_res_arr_mem_19169"))
            return 1;
        
        int32_t num_threads_19171 = num_groups_18243 * segred_group_sizze_18241;
        
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_18248, 0,
                                                sizeof(bool), NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_18248, 1,
                                                sizeof(bool) *
                                                segred_group_sizze_18241,
                                                NULL));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_18248, 2,
                                                sizeof(iota_arg_17470),
                                                &iota_arg_17470));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_18248, 3,
                                                sizeof(num_groups_18243),
                                                &num_groups_18243));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_18248, 4,
                                                sizeof(mem_18364.mem),
                                                &mem_18364.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_18248, 5,
                                                sizeof(mem_18412.mem),
                                                &mem_18412.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_18248, 6,
                                                sizeof(counter_mem_19167.mem),
                                                &counter_mem_19167.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_18248, 7,
                                                sizeof(group_res_arr_mem_19169.mem),
                                                &group_res_arr_mem_19169.mem));
        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->segred_nonseg_18248, 8,
                                                sizeof(num_threads_19171),
                                                &num_threads_19171));
        if (1 * (num_groups_18243 * segred_group_sizze_18241) != 0) {
            const size_t global_work_sizze_19379[1] = {num_groups_18243 *
                         segred_group_sizze_18241};
            const size_t local_work_sizze_19383[1] = {segred_group_sizze_18241};
            int64_t time_start_19380 = 0, time_end_19381 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with global work size [",
                        "segred_nonseg_18248");
                fprintf(stderr, "%zu", global_work_sizze_19379[0]);
                fprintf(stderr, "] and local work size [");
                fprintf(stderr, "%zu", local_work_sizze_19383[0]);
                fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                        (int) (0 + sizeof(bool) + sizeof(bool) *
                               segred_group_sizze_18241));
                time_start_19380 = get_wall_time();
            }
            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                            ctx->segred_nonseg_18248,
                                                            1, NULL,
                                                            global_work_sizze_19379,
                                                            local_work_sizze_19383,
                                                            0, NULL,
                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                            &ctx->segred_nonseg_18248_runs,
                                                                                                            &ctx->segred_nonseg_18248_total_runtime)));
            if (ctx->debugging) {
                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                time_end_19381 = get_wall_time();
                
                long time_diff_19382 = time_end_19381 - time_start_19380;
                
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "segred_nonseg_18248", time_diff_19382);
            }
        }
        
        bool read_res_19384;
        
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     mem_18412.mem, CL_TRUE, 0 *
                                                     sizeof(bool), sizeof(bool),
                                                     &read_res_19384, 0, NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_scalar_from_dev_runs,
                                                                                                     &ctx->copy_scalar_from_dev_total_runtime)));
        
        bool res_17811 = read_res_19384;
        int32_t res_17820 = 1 + count_17495;
        bool loop_cond_17821 = !res_17811;
        int32_t sizze_tmp_18503 = partition_sizze_17800;
        struct memblock_device shp_mem_tmp_18504;
        
        shp_mem_tmp_18504.references = NULL;
        if (memblock_set_device(ctx, &shp_mem_tmp_18504, &mem_18409,
                                "mem_18409") != 0)
            return 1;
        
        struct memblock_device arr_mem_tmp_18505;
        
        arr_mem_tmp_18505.references = NULL;
        if (memblock_set_device(ctx, &arr_mem_tmp_18505, &mem_18364,
                                "mem_18364") != 0)
            return 1;
        
        bool loop_while_tmp_18506 = loop_cond_17821;
        bool stop_tmp_18509 = res_17811;
        int32_t count_tmp_18510;
        
        count_tmp_18510 = res_17820;
        sizze_17490 = sizze_tmp_18503;
        if (memblock_set_device(ctx, &shp_mem_18302, &shp_mem_tmp_18504,
                                "shp_mem_tmp_18504") != 0)
            return 1;
        if (memblock_set_device(ctx, &arr_mem_18303, &arr_mem_tmp_18505,
                                "arr_mem_tmp_18505") != 0)
            return 1;
        loop_while_17491 = loop_while_tmp_18506;
        stop_17494 = stop_tmp_18509;
        count_17495 = count_tmp_18510;
        if (memblock_unref_device(ctx, &arr_mem_tmp_18505,
                                  "arr_mem_tmp_18505") != 0)
            return 1;
        if (memblock_unref_device(ctx, &shp_mem_tmp_18504,
                                  "shp_mem_tmp_18504") != 0)
            return 1;
        if (memblock_unref_device(ctx, &group_res_arr_mem_19169,
                                  "group_res_arr_mem_19169") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18409, "mem_18409") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18406, "mem_18406") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18403, "mem_18403") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18399, "mem_18399") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18395, "mem_18395") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18384, "mem_18384") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18381, "mem_18381") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18378, "mem_18378") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18374, "mem_18374") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18371, "mem_18371") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18368, "mem_18368") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18364, "mem_18364") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18361, "mem_18361") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18358, "mem_18358") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18354, "mem_18354") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18351, "mem_18351") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18348, "mem_18348") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18345, "mem_18345") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18342, "mem_18342") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18339, "mem_18339") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18330, "mem_18330") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18327, "mem_18327") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18324, "mem_18324") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18320, "mem_18320") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18317, "mem_18317") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18313, "mem_18313") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18310, "mem_18310") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_18307, "mem_18307") != 0)
            return 1;
    }
    sizze_17484 = sizze_17490;
    if (memblock_set_device(ctx, &res_mem_18413, &shp_mem_18302,
                            "shp_mem_18302") != 0)
        return 1;
    if (memblock_set_device(ctx, &res_mem_18414, &arr_mem_18303,
                            "arr_mem_18303") != 0)
        return 1;
    res_17485 = loop_while_17491;
    res_17488 = stop_17494;
    res_17489 = count_17495;
    if (memblock_unref_device(ctx, &mem_18298, "mem_18298") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_18333, "mem_18333") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_18335, "mem_18335") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_18412, "mem_18412") != 0)
        return 1;
    out_arrsizze_18465 = n_17467;
    if (memblock_set_device(ctx, &out_mem_18464, &res_mem_18414,
                            "res_mem_18414") != 0)
        return 1;
    (*out_mem_p_19193).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_19193, &out_mem_18464,
                            "out_mem_18464") != 0)
        return 1;
    *out_out_arrsizze_19194 = out_arrsizze_18465;
    if (memblock_unref_device(ctx, &arr_mem_18303, "arr_mem_18303") != 0)
        return 1;
    if (memblock_unref_device(ctx, &shp_mem_18302, "shp_mem_18302") != 0)
        return 1;
    if (memblock_unref_device(ctx, &res_mem_18414, "res_mem_18414") != 0)
        return 1;
    if (memblock_unref_device(ctx, &res_mem_18413, "res_mem_18413") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_18412, "mem_18412") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_18335, "mem_18335") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_18333, "mem_18333") != 0)
        return 1;
    if (memblock_unref_device(ctx, &group_res_arr_mem_18478,
                              "group_res_arr_mem_18478") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_18301, "mem_18301") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_18298, "mem_18298") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_18464, "out_mem_18464") != 0)
        return 1;
    return 0;
}
static int futrts__map_transpose_i32(struct futhark_context *ctx,
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
            if (in_elems_7 * sizeof(int32_t) > 0) {
                OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                             srcmem_2.mem,
                                                             destmem_0.mem,
                                                             srcoffset_3,
                                                             destoffset_1,
                                                             in_elems_7 *
                                                             sizeof(int32_t), 0,
                                                             NULL,
                                                             ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                             &ctx->copy_dev_to_dev_runs,
                                                                                                             &ctx->copy_dev_to_dev_total_runtime)));
                if (ctx->debugging)
                    OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            }
        } else {
            if (sle32(x_elems_5, 8) && slt32(16, y_elems_6)) {
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_width,
                                                        0, 1088, NULL));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_width,
                                                        1, sizeof(destoffset_1),
                                                        &destoffset_1));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_width,
                                                        2, sizeof(srcoffset_3),
                                                        &srcoffset_3));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_width,
                                                        3, sizeof(num_arrays_4),
                                                        &num_arrays_4));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_width,
                                                        4, sizeof(x_elems_5),
                                                        &x_elems_5));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_width,
                                                        5, sizeof(y_elems_6),
                                                        &y_elems_6));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_width,
                                                        6, sizeof(in_elems_7),
                                                        &in_elems_7));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_width,
                                                        7, sizeof(out_elems_8),
                                                        &out_elems_8));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_width,
                                                        8, sizeof(mulx_9),
                                                        &mulx_9));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_width,
                                                        9, sizeof(muly_10),
                                                        &muly_10));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_width,
                                                        10,
                                                        sizeof(destmem_0.mem),
                                                        &destmem_0.mem));
                OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_width,
                                                        11,
                                                        sizeof(srcmem_2.mem),
                                                        &srcmem_2.mem));
                if (1 * (squot32(x_elems_5 + 16 - 1, 16) * 16) *
                    (squot32(squot32(y_elems_6 + muly_10 - 1, muly_10) + 16 - 1,
                             16) * 16) * (num_arrays_4 * 1) != 0) {
                    const size_t global_work_sizze_19385[3] =
                                 {squot32(x_elems_5 + 16 - 1, 16) * 16,
                                  squot32(squot32(y_elems_6 + muly_10 - 1,
                                                  muly_10) + 16 - 1, 16) * 16,
                                  num_arrays_4 * 1};
                    const size_t local_work_sizze_19389[3] = {16, 16, 1};
                    int64_t time_start_19386 = 0, time_end_19387 = 0;
                    
                    if (ctx->debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "map_transpose_i32_low_width");
                        fprintf(stderr, "%zu", global_work_sizze_19385[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_19385[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_19385[2]);
                        fprintf(stderr, "] and local work size [");
                        fprintf(stderr, "%zu", local_work_sizze_19389[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_19389[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_19389[2]);
                        fprintf(stderr,
                                "]; local memory parameters sum to %d bytes.\n",
                                (int) (0 + 1088));
                        time_start_19386 = get_wall_time();
                    }
                    OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                    ctx->map_transpose_i32_low_width,
                                                                    3, NULL,
                                                                    global_work_sizze_19385,
                                                                    local_work_sizze_19389,
                                                                    0, NULL,
                                                                    ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                    &ctx->map_transpose_i32_low_width_runs,
                                                                                                                    &ctx->map_transpose_i32_low_width_total_runtime)));
                    if (ctx->debugging) {
                        OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                        time_end_19387 = get_wall_time();
                        
                        long time_diff_19388 = time_end_19387 -
                             time_start_19386;
                        
                        fprintf(stderr, "kernel %s runtime: %ldus\n",
                                "map_transpose_i32_low_width", time_diff_19388);
                    }
                }
            } else {
                if (sle32(y_elems_6, 8) && slt32(16, x_elems_5)) {
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_height,
                                                            0, 1088, NULL));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_height,
                                                            1,
                                                            sizeof(destoffset_1),
                                                            &destoffset_1));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_height,
                                                            2,
                                                            sizeof(srcoffset_3),
                                                            &srcoffset_3));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_height,
                                                            3,
                                                            sizeof(num_arrays_4),
                                                            &num_arrays_4));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_height,
                                                            4,
                                                            sizeof(x_elems_5),
                                                            &x_elems_5));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_height,
                                                            5,
                                                            sizeof(y_elems_6),
                                                            &y_elems_6));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_height,
                                                            6,
                                                            sizeof(in_elems_7),
                                                            &in_elems_7));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_height,
                                                            7,
                                                            sizeof(out_elems_8),
                                                            &out_elems_8));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_height,
                                                            8, sizeof(mulx_9),
                                                            &mulx_9));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_height,
                                                            9, sizeof(muly_10),
                                                            &muly_10));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_height,
                                                            10,
                                                            sizeof(destmem_0.mem),
                                                            &destmem_0.mem));
                    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_low_height,
                                                            11,
                                                            sizeof(srcmem_2.mem),
                                                            &srcmem_2.mem));
                    if (1 * (squot32(squot32(x_elems_5 + mulx_9 - 1, mulx_9) +
                                     16 - 1, 16) * 16) * (squot32(y_elems_6 +
                                                                  16 - 1, 16) *
                                                          16) * (num_arrays_4 *
                                                                 1) != 0) {
                        const size_t global_work_sizze_19390[3] =
                                     {squot32(squot32(x_elems_5 + mulx_9 - 1,
                                                      mulx_9) + 16 - 1, 16) *
                                      16, squot32(y_elems_6 + 16 - 1, 16) * 16,
                                      num_arrays_4 * 1};
                        const size_t local_work_sizze_19394[3] = {16, 16, 1};
                        int64_t time_start_19391 = 0, time_end_19392 = 0;
                        
                        if (ctx->debugging) {
                            fprintf(stderr,
                                    "Launching %s with global work size [",
                                    "map_transpose_i32_low_height");
                            fprintf(stderr, "%zu", global_work_sizze_19390[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_19390[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_19390[2]);
                            fprintf(stderr, "] and local work size [");
                            fprintf(stderr, "%zu", local_work_sizze_19394[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_19394[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_19394[2]);
                            fprintf(stderr,
                                    "]; local memory parameters sum to %d bytes.\n",
                                    (int) (0 + 1088));
                            time_start_19391 = get_wall_time();
                        }
                        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                        ctx->map_transpose_i32_low_height,
                                                                        3, NULL,
                                                                        global_work_sizze_19390,
                                                                        local_work_sizze_19394,
                                                                        0, NULL,
                                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                        &ctx->map_transpose_i32_low_height_runs,
                                                                                                                        &ctx->map_transpose_i32_low_height_total_runtime)));
                        if (ctx->debugging) {
                            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                            time_end_19392 = get_wall_time();
                            
                            long time_diff_19393 = time_end_19392 -
                                 time_start_19391;
                            
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "map_transpose_i32_low_height",
                                    time_diff_19393);
                        }
                    }
                } else {
                    if (sle32(x_elems_5, 8) && sle32(y_elems_6, 8)) {
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_small,
                                                                0, 1, NULL));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_small,
                                                                1,
                                                                sizeof(destoffset_1),
                                                                &destoffset_1));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_small,
                                                                2,
                                                                sizeof(srcoffset_3),
                                                                &srcoffset_3));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_small,
                                                                3,
                                                                sizeof(num_arrays_4),
                                                                &num_arrays_4));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_small,
                                                                4,
                                                                sizeof(x_elems_5),
                                                                &x_elems_5));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_small,
                                                                5,
                                                                sizeof(y_elems_6),
                                                                &y_elems_6));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_small,
                                                                6,
                                                                sizeof(in_elems_7),
                                                                &in_elems_7));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_small,
                                                                7,
                                                                sizeof(out_elems_8),
                                                                &out_elems_8));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_small,
                                                                8,
                                                                sizeof(mulx_9),
                                                                &mulx_9));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_small,
                                                                9,
                                                                sizeof(muly_10),
                                                                &muly_10));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_small,
                                                                10,
                                                                sizeof(destmem_0.mem),
                                                                &destmem_0.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32_small,
                                                                11,
                                                                sizeof(srcmem_2.mem),
                                                                &srcmem_2.mem));
                        if (1 * (squot32(num_arrays_4 * x_elems_5 * y_elems_6 +
                                         256 - 1, 256) * 256) != 0) {
                            const size_t global_work_sizze_19395[1] =
                                         {squot32(num_arrays_4 * x_elems_5 *
                                                  y_elems_6 + 256 - 1, 256) *
                                         256};
                            const size_t local_work_sizze_19399[1] = {256};
                            int64_t time_start_19396 = 0, time_end_19397 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "map_transpose_i32_small");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_19395[0]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_19399[0]);
                                fprintf(stderr,
                                        "]; local memory parameters sum to %d bytes.\n",
                                        (int) (0 + 1));
                                time_start_19396 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->map_transpose_i32_small,
                                                                            1,
                                                                            NULL,
                                                                            global_work_sizze_19395,
                                                                            local_work_sizze_19399,
                                                                            0,
                                                                            NULL,
                                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                            &ctx->map_transpose_i32_small_runs,
                                                                                                                            &ctx->map_transpose_i32_small_total_runtime)));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_19397 = get_wall_time();
                                
                                long time_diff_19398 = time_end_19397 -
                                     time_start_19396;
                                
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "map_transpose_i32_small",
                                        time_diff_19398);
                            }
                        }
                    } else {
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32,
                                                                0, 4224, NULL));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32,
                                                                1,
                                                                sizeof(destoffset_1),
                                                                &destoffset_1));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32,
                                                                2,
                                                                sizeof(srcoffset_3),
                                                                &srcoffset_3));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32,
                                                                3,
                                                                sizeof(num_arrays_4),
                                                                &num_arrays_4));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32,
                                                                4,
                                                                sizeof(x_elems_5),
                                                                &x_elems_5));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32,
                                                                5,
                                                                sizeof(y_elems_6),
                                                                &y_elems_6));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32,
                                                                6,
                                                                sizeof(in_elems_7),
                                                                &in_elems_7));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32,
                                                                7,
                                                                sizeof(out_elems_8),
                                                                &out_elems_8));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32,
                                                                8,
                                                                sizeof(mulx_9),
                                                                &mulx_9));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32,
                                                                9,
                                                                sizeof(muly_10),
                                                                &muly_10));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32,
                                                                10,
                                                                sizeof(destmem_0.mem),
                                                                &destmem_0.mem));
                        OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->map_transpose_i32,
                                                                11,
                                                                sizeof(srcmem_2.mem),
                                                                &srcmem_2.mem));
                        if (1 * (squot32(x_elems_5 + 32 - 1, 32) * 32) *
                            (squot32(y_elems_6 + 32 - 1, 32) * 8) *
                            (num_arrays_4 * 1) != 0) {
                            const size_t global_work_sizze_19400[3] =
                                         {squot32(x_elems_5 + 32 - 1, 32) * 32,
                                          squot32(y_elems_6 + 32 - 1, 32) * 8,
                                          num_arrays_4 * 1};
                            const size_t local_work_sizze_19404[3] = {32, 8, 1};
                            int64_t time_start_19401 = 0, time_end_19402 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "map_transpose_i32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_19400[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_19400[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_19400[2]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_19404[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_19404[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_19404[2]);
                                fprintf(stderr,
                                        "]; local memory parameters sum to %d bytes.\n",
                                        (int) (0 + 4224));
                                time_start_19401 = get_wall_time();
                            }
                            OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                            ctx->map_transpose_i32,
                                                                            3,
                                                                            NULL,
                                                                            global_work_sizze_19400,
                                                                            local_work_sizze_19404,
                                                                            0,
                                                                            NULL,
                                                                            ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                                            &ctx->map_transpose_i32_runs,
                                                                                                                            &ctx->map_transpose_i32_total_runtime)));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
                                time_end_19402 = get_wall_time();
                                
                                long time_diff_19403 = time_end_19402 -
                                     time_start_19401;
                                
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "map_transpose_i32", time_diff_19403);
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
                                 struct memblock_device mem_18934,
                                 int32_t num_elems_18935, float val_18936)
{
    int32_t group_sizze_18941;
    
    group_sizze_18941 = ctx->sizes.mainzigroup_sizze_18941;
    
    int32_t num_groups_18942;
    
    num_groups_18942 = squot32(num_elems_18935 +
                               sext_i32_i32(group_sizze_18941) - 1,
                               sext_i32_i32(group_sizze_18941));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_18938, 0,
                                            sizeof(mem_18934.mem),
                                            &mem_18934.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_18938, 1,
                                            sizeof(num_elems_18935),
                                            &num_elems_18935));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_18938, 2,
                                            sizeof(val_18936), &val_18936));
    if (1 * (num_groups_18942 * group_sizze_18941) != 0) {
        const size_t global_work_sizze_19405[1] = {num_groups_18942 *
                     group_sizze_18941};
        const size_t local_work_sizze_19409[1] = {group_sizze_18941};
        int64_t time_start_19406 = 0, time_end_19407 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "replicate_18938");
            fprintf(stderr, "%zu", global_work_sizze_19405[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_19409[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_19406 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->replicate_18938, 1,
                                                        NULL,
                                                        global_work_sizze_19405,
                                                        local_work_sizze_19409,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->replicate_18938_runs,
                                                                                                        &ctx->replicate_18938_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_19407 = get_wall_time();
            
            long time_diff_19408 = time_end_19407 - time_start_19406;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "replicate_18938",
                    time_diff_19408);
        }
    }
    return 0;
}
static int futrts__replicate_i32(struct futhark_context *ctx,
                                 struct memblock_device mem_18466,
                                 int32_t num_elems_18467, int32_t val_18468)
{
    int32_t group_sizze_18473;
    
    group_sizze_18473 = ctx->sizes.mainzigroup_sizze_18473;
    
    int32_t num_groups_18474;
    
    num_groups_18474 = squot32(num_elems_18467 +
                               sext_i32_i32(group_sizze_18473) - 1,
                               sext_i32_i32(group_sizze_18473));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_18470, 0,
                                            sizeof(mem_18466.mem),
                                            &mem_18466.mem));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_18470, 1,
                                            sizeof(num_elems_18467),
                                            &num_elems_18467));
    OPENCL_SUCCEED_OR_RETURN(clSetKernelArg(ctx->replicate_18470, 2,
                                            sizeof(val_18468), &val_18468));
    if (1 * (num_groups_18474 * group_sizze_18473) != 0) {
        const size_t global_work_sizze_19410[1] = {num_groups_18474 *
                     group_sizze_18473};
        const size_t local_work_sizze_19414[1] = {group_sizze_18473};
        int64_t time_start_19411 = 0, time_end_19412 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "replicate_18470");
            fprintf(stderr, "%zu", global_work_sizze_19410[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_19414[0]);
            fprintf(stderr, "]; local memory parameters sum to %d bytes.\n",
                    (int) 0);
            time_start_19411 = get_wall_time();
        }
        OPENCL_SUCCEED_OR_RETURN(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                        ctx->replicate_18470, 1,
                                                        NULL,
                                                        global_work_sizze_19410,
                                                        local_work_sizze_19414,
                                                        0, NULL,
                                                        ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                        &ctx->replicate_18470_runs,
                                                                                                        &ctx->replicate_18470_total_runtime)));
        if (ctx->debugging) {
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
            time_end_19412 = get_wall_time();
            
            long time_diff_19413 = time_end_19412 - time_start_19411;
            
            fprintf(stderr, "kernel %s runtime: %ldus\n", "replicate_18470",
                    time_diff_19413);
        }
    }
    return 0;
}
struct futhark_f32_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    if (dim0 * sizeof(float) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueWriteBuffer(ctx->opencl.queue,
                                                      arr->mem.mem, CL_TRUE, 0,
                                                      dim0 * sizeof(float),
                                                      data + 0, 0, NULL,
                                                      ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                      &ctx->copy_dev_to_host_runs,
                                                                                                      &ctx->copy_dev_to_host_total_runtime)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              cl_mem data, int offset,
                                              int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    if (dim0 * sizeof(float) > 0) {
        OPENCL_SUCCEED_OR_RETURN(clEnqueueCopyBuffer(ctx->opencl.queue, data,
                                                     arr->mem.mem, offset, 0,
                                                     dim0 * sizeof(float), 0,
                                                     NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_dev_to_dev_runs,
                                                                                                     &ctx->copy_dev_to_dev_total_runtime)));
        if (ctx->debugging)
            OPENCL_SUCCEED_FATAL(clFinish(ctx->opencl.queue));
    }
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data)
{
    lock_lock(&ctx->lock);
    if (arr->shape[0] * sizeof(float) > 0)
        OPENCL_SUCCEED_OR_RETURN(clEnqueueReadBuffer(ctx->opencl.queue,
                                                     arr->mem.mem, CL_TRUE, 0,
                                                     arr->shape[0] *
                                                     sizeof(float), data + 0, 0,
                                                     NULL,
                                                     ctx->profiling_paused ? NULL : opencl_get_event(&ctx->opencl,
                                                                                                     &ctx->copy_host_to_dev_runs,
                                                                                                     &ctx->copy_host_to_dev_total_runtime)));
    lock_unlock(&ctx->lock);
    return 0;
}
cl_mem futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                 struct futhark_f32_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr)
{
    (void) ctx;
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_1d **out0, const
                       struct futhark_f32_1d *in0)
{
    struct memblock_device arr_mem_18295;
    
    arr_mem_18295.references = NULL;
    
    int32_t n_17467;
    struct memblock_device out_mem_18464;
    
    out_mem_18464.references = NULL;
    
    int32_t out_arrsizze_18465;
    
    lock_lock(&ctx->lock);
    arr_mem_18295 = in0->mem;
    n_17467 = in0->shape[0];
    
    int ret = futrts_main(ctx, &out_mem_18464, &out_arrsizze_18465,
                          arr_mem_18295, n_17467);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_18464;
        (*out0)->shape[0] = out_arrsizze_18465;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
